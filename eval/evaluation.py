import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression,  LinearRegression
from sklearn.preprocessing import StandardScaler

def neighbor_kept_ratio_eval(fold1, fold2,n_neighbors=30):

    # make sure both fold1 and fold2 is numpy object
    # fold index: unseen on training set

    nn1= NearestNeighbors(n_neighbors=n_neighbors+1)
    nn2 = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn1.fit(fold1)
    nn2.fit(fold2)
    graph_nn1 = nn1.kneighbors_graph(fold1).toarray()
    graph_nn1 -= np.eye(fold1.shape[0]) # Removing diagonal
    graph_nn2 = nn2.kneighbors_graph(fold2).toarray()
    graph_nn2 -= np.eye(fold2.shape[0]) # Removing diagonal
    
    neighbor_kept = np.sum((graph_nn1 * graph_nn2), axis = 0)
    neighbor_kept_ratio = np.mean(np.divide(neighbor_kept,n_neighbors))
    return neighbor_kept_ratio


def cca_dist(emb1, emb2, sample_p = 0.7):
    scaler = StandardScaler()
    H1 = scaler.fit_transform(emb1)
    H2 = scaler.fit_transform(emb2)
    indices = np.random.choice(emb1.shape[0], int(emb1.shape[0]*sample_p), replace=False)
    n = H1.shape[1]
    
    # Compute inverse square roots
    inv1 = scipy.linalg.sqrtm(H1[indices,:].T@H1[indices,:]/n)
    #inv1 = scipy.linalg.sqrtm(H1.T@H2/n)
    inv2 = scipy.linalg.sqrtm(H2[indices,:].T@H2[indices,:]/n)

    cov = (H1[indices,:] @ inv1).T @ (H2[indices,:] @ inv2)/n
    #cov = (H1 @ inv1).T @ (H2 @ inv2)/n
    U, _, Vt = scipy.linalg.svd(cov, full_matrices=False)
    
    return np.linalg.norm(H1[-indices,:] @ inv1 @ U - H2[-indices,:] @ inv2 @ Vt.T, 'fro')


def linear_classifier(X, label, p=0.7, logistic=True):
    # Split the dataset into train and test sets
    train_size = int(p * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    if logistic:
        # Logistic Regression for classification
        clf = LogisticRegression().fit(X[train_idx], label[train_idx])
        score = clf.score(X[test_idx], label[test_idx])
    else:
        # Linear Regression for regression
        reg = LinearRegression().fit(X[train_idx], label[train_idx])
        score = reg.score(X[test_idx], label[test_idx])
    return score

def acc(fold1, fold2, label, n_comp = 3):
    # Fit a Gaussian mixture with EM using five components
    gmm = GaussianMixture(n_components=n_comp, random_state=0).fit(fold1)
    labels = np.argmax(np.array(gmm.predict_proba(fold2)),axis = 1)
    acc = normalized_mutual_info_score(labels, label, average_method='arithmetic')    
    # acc = hungarian_acc(np.array(label, dtype ='int'),labels)
    return acc

def evaluate_CV(fold1, fold2, out, label=None, n_comp = 3, clf = True):

    result = {}
    test_clusters = [2,3,5,7,10]
    for i, num_cluster in enumerate([value for index, value in enumerate(test_clusters) if value < fold1.shape[0]]):
        knn1 = KMeans(n_clusters= num_cluster,n_init = 'auto').fit(fold1)
        knn2 = KMeans(n_clusters= num_cluster,n_init = 'auto').fit(fold2)
        gmm1 = GaussianMixture(n_components=num_cluster, init_params = 'k-means++').fit(fold1)
        label1 = np.argmax(np.array(gmm1.predict_proba(fold1)),axis = 1)
        gmm2 = GaussianMixture(n_components=num_cluster, init_params = 'k-means++').fit(fold2)
        label2 = np.argmax(np.array(gmm2.predict_proba(fold2)),axis = 1)
        # label_mapping = match_labels(knn1.labels_, knn2.labels_)
        # matched_labels2 = label_mapping[knn2.labels_-1]   # Adjust for zero-indexing
        # accuracy = np.mean(knn1.labels_ == matched_labels2)
        result['knn_ari'+ str(num_cluster)] = adjusted_rand_score(knn1.labels_, knn2.labels_)
        result['knn_nmi'+ str(num_cluster)] = normalized_mutual_info_score(knn1.labels_, knn2.labels_, average_method='arithmetic')
        result['gmm_ari'+ str(num_cluster)] = adjusted_rand_score(label1, label2)
        result['gmm_nmi'+ str(num_cluster)] = normalized_mutual_info_score(label1, label2, average_method='arithmetic')

    h1 = linkage(fold1, method = 'ward')
    h2 = linkage(fold2, method = 'ward')
    result['cluster_spr'] = spearmanr(h1.flatten(),h2.flatten())[0]

    test_neighbors = [1, 3, 5, 10, 20, 30, 50]
    for i, num_neighbor in enumerate([value for index, value in enumerate(test_neighbors) if value < fold1.shape[0]]):
        result['neighbor_kept_' + str(num_neighbor)] = float(neighbor_kept_ratio_eval(fold1, fold2, n_neighbors=num_neighbor))
    
    result['cca_dist'] = cca_dist(fold1, fold2)

    # clf = LogisticRegression().fit(fold2, label)
    # result['acc'] = clf.score(fold1, label)
    #if fold_index is None:
    #    indices = np.arange(fold1.shape[0])
    #    train_size = int(fold1.shape[0]*0.7)
    #    np.random.shuffle(indices)
    #    clf2 = LogisticRegression().fit(out[indices[:train_size]], label[:train_size])
    #    result['pred_acc'] = clf2.score(out[indices[train_size:]], label[indices[train_size:]])
    #else:
    #    clf2 = LogisticRegression().fit(out[fold_index[0]], label[fold_index[0]])
    #    result['pred_acc'] = clf2.score(out[fold_index[1]], label[fold_index[1]])

    # Fit a Gaussian mixture with EM using five components
    # gmm = GaussianMixture(n_components=n_comp, random_state=0).fit(fold1)
    # labels = gmm.predict(fold2)
    # result['acc'] = hungarian_acc(np.array(label, dtype ='int'),labels)
    result['acc'] = linear_classifier(out, label, logistic = clf)
    return result

