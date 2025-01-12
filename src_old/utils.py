"""
    Description: This file contains utility functions that are used in the main code

    Author : Gallegos Carvajal Ian Marco
"""
import numpy as np

def vis_save_image_pdf(visualizer, image_out):
    visualizer.show('results/img/' + image_out + '.png')
    print(f"SAVED: results/img/{image_out}.png")
    visualizer.show('results/img/pdf/' + image_out + '.pdf')
    print(f"SAVED: results/img/pdf/{image_out}.pdf")

def plt_save_image_pdf(plt, image_out):
    plt.savefig('results/img/' + image_out + '.png')
    print(f"SAVED: results/img/{image_out}.png")
    plt.savefig('results/img/pdf/' + image_out + '.pdf')
    print(f"SAVED: results/img/pdf/{image_out}.pdf")

def set_feature_np(df, feature):
    if feature == 'ada_embedding':
        for i in range(len(df)):
            df[feature].iloc[i] = np.fromstring(df[feature].iloc[i][1:-1], sep=', ')
    elif feature == 'ada_embedding_normalized':
        for i in range(len(df)):
            df[feature].iloc[i] = np.fromstring(df[feature].iloc[i][1:-1], sep=' ')
    else:
        for i in range(len(df)):
            df[feature].iloc[i] = np.fromstring(df[feature].iloc[i][1:-1], sep=', ')
