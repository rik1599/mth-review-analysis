import argparse
import base64
import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm


class GPTQuery:
    def _init_(
        self,
        log_dir,
        api_key_path,
        system_prompt_path,
    ):
        self.log_dir = log_dir
        self.api_key_path = api_key_path
        self.system_prompt_path = system_prompt_path

        self.logger = self.setup_logging(self.log_dir)
        self.api_key = self.load_api_key(self.api_key_path)
        self.logger.info(f"Loaded API key: {'*' * 4}{self.api_key[-4:]})")
        self.system_prompt = self.load_system_prompt(self.system_prompt_path)
        self.logger.info(f"Loaded system prompt: {self.system_prompt[:50]}...")

    def setup_logging(self, log_dir):
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        log_file = os.path.join(
            log_dir,
            f"{script_name}{datetime.now().strftime('%Y%m%d%H%M%S')}.log",
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file)],
        )
        return logging.getLogger()

    def load_api_key(self, api_key_path):
        with open(api_key_path, "r") as f:
            return json.load(f)["api_key"]

    def load_system_prompt(self, prompt_path):
        with open(prompt_path, "r") as f:
            return f.read().strip()

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_content(self, caption, image_path):
        base64_image = self.encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reasoning_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "post_description": {
                                "type": "string",
                                "description": "Description of the post considering both the text and the image. Explain the main intention of the author and the context of the event.",
                            },
                            "search_query": {
                                "type": "string",
                                "description": "Single, precise search query based on the author's main intention to find more context and relevant articles for the post.",
                            },
                            "general_topic": {
                                "type": "string",
                                "description": "A broader topic or theme that encompasses the main focus of the post.",
                            },
                        },
                        "required": [
                            "post_description",
                            "search_query",
                            "general_topic",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 1000,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        return response.json()

    def process_data(self, input_file, image_dir, output_dir):
        df = pd.read_csv(input_file)
        self.logger.info(f"Loaded {len(df)} rows from {input_file}")

        # Load existing results if available
        results_file = os.path.join(output_dir, "gpt_results.csv")
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            processed_ids = results_df["id"].unique()
            self.logger.info(
                f"Loaded {len(processed_ids)} processed ids from existing results"
            )
        else:
            processed_ids = []

        df = df[~df["id"].isin(processed_ids)]
        self.logger.info(f"Posts left to process: {len(df)}")

        results = []
        request_count_today = 0
        last_request_time = datetime.now() - timedelta(minutes=1)

        for index, row in tqdm(df.iterrows(), total=len(df)):
            # rate limiting (3 RPM) ==> got upgrated, don't know the new limit
            while (datetime.now() - last_request_time).seconds < 0.2:
                time.sleep(1)

            # daily limit (200 RPD) ==> got upgrated, don't know the new limit
            if request_count_today >= 6000:
                self.logger.info("Daily processing limit reached. Stopping.")
                break

            image_path = os.path.join(image_dir, f"{row['id']}.jpg")
            if not os.path.exists(image_path):
                self.logger.warning(f"Image not found for id {row['id']}")
                continue

            try:
                response = self.analyze_content(row["caption"], image_path)
                content = response["choices"][0]["message"]["content"]
                content_dict = json.loads(content)
                results.append(
                    {
                        "id": row["id"],
                        "post_description": content_dict["post_description"],
                        "search_query": content_dict["search_query"],
                        "general_topic": content_dict["general_topic"],
                    }
                )

                if (index + 1) % 50 == 0 or index == len(df) - 1:
                    self.save_results(results, output_dir)
                    results = []

            except Exception as e:
                self.logger.error(
                    f"Error processing row {index} (id: {row['id']}): {str(e)}"
                )
                self.logger.info(f"model response: {response}")

            request_count_today += 1
            last_request_time = datetime.now()

        if results:
            self.save_results(results, output_dir)

    def save_results(self, results, output_dir):
        output_file = os.path.join(output_dir, "gpt_results.csv")

        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            results_df = pd.DataFrame(results)
            combined_df = pd.concat([existing_df, results_df])
            combined_df.drop_duplicates(subset=["id"], inplace=True)
            combined_df.to_csv(output_file, index=False)
            self.logger.info(f"Appended {len(results)} results to {output_file}")
            self.logger.info(
                f"Total results saved: {len(combined_df)}" + "\n" + "-" * 50
            )
        else:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "id",
                        "post_description",
                        "search_query",
                        "general_topic",
                    ],
                )
                writer.writeheader()
                writer.writerows(results)
            self.logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze social media content")
    parser.add_argument(
        "--input-file", required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Path to the directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the output directory for saving results",
    )
    parser.add_argument(
        "--log-dir", required=True, help="Path to the directory for saving log files"
    )
    parser.add_argument(
        "--api-key-path",
        required=True,
        help="Path to the JSON file containing the API key",
    )
    parser.add_argument(
        "--system-prompt-path",
        required=True,
        help="Path to the file containing the system prompt",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    gpt_query = GPTQuery(
        log_dir=args.log_dir,
        api_key_path=args.api_key_path,
        system_prompt_path=args.system_prompt_path,
    )

    gpt_query.process_data(
        input_file=args.input_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
    )


if __name__ == "_main_":
    main()