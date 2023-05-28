from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(X, label=None, name=0):
    color_list = ['rosybrown', 'firebrick', 'coral', 'orange', 'gold', 'olive', 'chartreuse', 'honeydew', 'palegreen', 'turquoise', 'teal', 'deepskyblue', 'dodgerblue', 'navy', 'slateblue','blueviolet','indigo','crimson']
    if label == None:
        label = np.arange(X.shape[0])
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    print(X_embedded)
    for _ in range(X_embedded.shape[0]):
        plt.scatter(X_embedded[_,0], X_embedded[_,1], c = color_list[label[_]], label = label[_])
    plt.legend()
    plt.savefig('tsne_%d.png'%name)

           
                  