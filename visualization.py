from sklearn import manifold, datasets

def norm(T):
    T_min,T_max = T.min(),T.max()
    return (T-T_min) / ((T_max-T_min ))

def tsne_visualization_1_view(embeddings):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(embeddings)
    colors = ['#FF0000', '#FF6600', '#FFFF00', '#99FF00', '#00FF00', '#00FFFF', '#0000FF', '#6600FF', '#FF00FF', '#FF0066']
    X_tsne = norm(X_tsne)
    plt.figure(figsize=(8, 8))
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(i // 100), color=colors[i % 10], fontdict={'weight': 'bold', 'size': 8})
    plt.show()


def tsne_visualization_2_view(embeddings):
    tsne = manifold.TSNE(n_components=1, init='pca', random_state=501)
    v1 = E[:, 0, :].reshape(embeddings.shape(0), -1)
    v2 = E[:, 1, :].reshape(embeddings.shape(0), -1)
    X_tsne = tsne.fit_transform(embeddings)
    colors = ['#FF0000', '#FF6600', '#FFFF00', '#99FF00', '#00FF00', '#00FFFF', '#0000FF', '#6600FF', '#FF00FF', '#FF0066']
    v1 = norm(v1)
    v2 = norm(v2)
    plt.figure(figsize=(16, 16))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("View 1", fontsize=22)
    plt.ylabel("View 2", fontsize=22)
    for i in range(v1.shape[0]):
        plt.text(v1[i], v2[i], str(i // 100), color=colors[i % 10], fontdict={'weight': 'bold', 'size': 15})
    plt.show()