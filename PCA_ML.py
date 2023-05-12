# =============================================================================
# ----------------------------PCA----------------------------------------------
# =============================================================================
print('PCA\n')

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

e = np.exp(1)
np.random.seed(4)


def pdf(x):
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x) + stats.norm(scale=4 / e).pdf(x))


y = np.random.normal(scale=0.5, size=(30000))
x = np.random.normal(scale=0.5, size=(30000))
z = np.random.normal(scale=0.1, size=len(x))

density = pdf(x) * pdf(y)
pdf_z = pdf(5 * z)

density *= pdf_z

a = x + y
b = 2 * y
c = a - b + z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm

a_mean = np.mean(a)
b_mean = np.mean(b)
c_mean = np.mean(c)

X = np.array(list(zip((a-a_mean),(b-b_mean),(c-c_mean))))

Xt = np.array(np.transpose(X))

P = (1/len(a))*(np.matmul(Xt,X))

eigenvalues, eigenvectors = np.linalg.eig(P)

eigenvectors = np.transpose(eigenvectors)
k = 3
top_k_eigenvectors = eigenvectors[:, :k]

# Projecting the data onto the top k eigenvectors
X_pca = np.dot(X,top_k_eigenvectors)

print('\nThe PCA components and figures using the Eigen decomposition available in numpy:\n')
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection="3d", elev=-40, azim=-80)
ax2 = fig.add_subplot(122, projection="3d", elev=30, azim=20)

ax1.scatter(a[::10], b[::10], c[::10], c=density[::10], marker="+", alpha=0.4)
ax2.scatter(a[::10], b[::10], c[::10], c=density[::10], marker="+", alpha=0.4)

V = top_k_eigenvectors.T
print(V)
x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
x_pca_plane.shape = (2, 2)
y_pca_plane.shape = (2, 2)
z_pca_plane.shape = (2, 2)
ax1.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
ax2.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)

ax1.set_title("View 1")
ax2.set_title("View 2")

plt.show()
print('\nThe PCA components and figures using the code available in the link:\n')

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401


def plot_figs(fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)
    ax.set_position([0, 0, 0.95, 1])

    ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker="+", alpha=0.4)
    Y = np.c_[a, b, c]

    pca = PCA(n_components=3)
    pca.fit(Y)
    
    V = pca.components_.T
    print(V)
    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
    x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)
    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])


elev = -40
azim = -80
plot_figs(1, elev, azim)

elev = 30
azim = 20
plot_figs(2, elev, azim)

plt.show()

print("\nAs we can clearly see that the results are exactly same. So yes we are getting the same result.")

