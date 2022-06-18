import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


def compare_models(models):
    def get_best_iter_in_100(df, colum_name, min=True):
        df = df[colum_name]
        indices = []
        elementos = []
        for i in range(0, round(len(df)/100)):
            if min:
                indice = df[i*100:(i+1)*100,].idxmin()
            else:
                indice = df[i*100:(i+1)*100,].idxmax()
            indices.append(indice)
            elementos.append(df.iloc[indice])
        return elementos, indices

    fig, ax= plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle(f"Comparación de modelos", fontsize=16)
    ax[0].set_title('Train Error')
    ax[1].set_title('Validation Error')
    ax[2].set_title('Train Accuracy')
    ax[3].set_title('Validation Accuracy')
    ax[0].set_ylim([0,1])
    ax[1].set_ylim([0,1])
    ax[2].set_ylim([0,1])
    ax[3].set_ylim([0,1])

    ax[0].set_xlabel('Iteraciones')
    ax[0].set_ylabel('Cross-entropy Error')

    ax[1].set_xlabel('Iteraciones')
    ax[1].set_ylabel('Cross-entropy Error')

    ax[2].set_xlabel('Iteraciones')
    ax[2].set_ylabel('Accuracy')
    ax[3].set_xlabel('Iteraciones')
    ax[2].set_ylabel('Accuracy')
    

    for model_name in models:
        df= pd.read_csv(f"experimentos/{model_name}/error_and_accuracy.csv",
                        header=None, names = ['Train error','Validation error', 'Train accuracy','Validation accuracy']) 

        train_error, train_error_indices = get_best_iter_in_100(df, "Train error")
        validation_error, validaton_error_indices = get_best_iter_in_100(df, "Validation error")

        ax[0].plot(train_error_indices,train_error)
        ax[1].plot(validaton_error_indices,validation_error)

        train_accuracy, train_accuracy_indices = get_best_iter_in_100(df, "Train accuracy", False)
        validation_accuracy, validaton_accuracy_indices = get_best_iter_in_100(df, "Validation accuracy", False)

        ax[2].plot(train_accuracy_indices,train_accuracy)
        ax[3].plot(validaton_accuracy_indices,validation_accuracy)



    ax[0].legend(models)
    ax[1].legend(models)
    ax[2].legend(models)
    ax[3].legend(models)


def plot_metrics(model_name):
    def get_best_iter_in_100(df, colum_name, min=True):
        df = df[colum_name]
        indices = []
        elementos = []
        for i in range(0, round(len(df)/100)):
            if min:
                indice = df[i*100:(i+1)*100,].idxmin()
            else:
                indice = df[i*100:(i+1)*100,].idxmax()
            indices.append(indice)
            elementos.append(df.iloc[indice])
        return elementos, indices

    df= pd.read_csv(f"experimentos/{model_name}/error_and_accuracy.csv",
                    header=None, names = ['Train error','Validation error', 'Train accuracy','Validation accuracy']) 

    train_error, train_error_indices = get_best_iter_in_100(df, "Train error")
    validation_error, validaton_error_indices = get_best_iter_in_100(df, "Validation error")
    fig, ax= plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(f"Modelo: {model_name}", fontsize=16)
    ax[0].set_title('Cross Entropy Error')
    ax[1].set_title('Accuracy')

    ax[0].set_ylim([0,1])
    ax[1].set_ylim([0,1])

    ax[0].plot(train_error_indices,train_error, c='r')
    ax[0].plot(validaton_error_indices,validation_error, c='b')
    ax[0].set_xlabel('Iteraciones')
    ax[0].set_ylabel('Cross-entropy Error')

    train_accuracy, train_accuracy_indices = get_best_iter_in_100(df, "Train accuracy", False)
    validation_accuracy, validaton_accuracy_indices = get_best_iter_in_100(df, "Validation accuracy", False)

    ax[1].plot(train_accuracy_indices,train_accuracy, c='r')
    ax[1].plot(validaton_accuracy_indices,validation_accuracy, c='b')
    ax[1].set_xlabel('Iteraciones')
    ax[1].set_ylabel('Accuracy')


    ax[0].legend(['Train error','Validation error'],loc='lower left')
    ax[1].legend(['Train acc','Validation acc'],loc='upper left')

    ax[0].text(0, 0.2, f"Train error: {round(min(train_error),2)} \n Validation error: {round(min(validation_error),2)}",
    bbox=dict(facecolor='white', alpha=0.2))

    ax[1].text(train_accuracy_indices[(len(train_accuracy_indices)//3)], 0.1,
    f"Train accuracy: {round(max(train_accuracy),2)} \n Validation accuracy: {round(max(validation_accuracy),2)}",
    bbox=dict(facecolor='white', alpha=0.2))