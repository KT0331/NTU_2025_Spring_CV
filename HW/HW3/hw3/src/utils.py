import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2*i, :] = np.array([u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0] * v[i][0], -u[i][1] * v[i][0], -v[i][0]])
        A[2*i+1, :] = np.array([0, 0, 0, u[i][0], u[i][1], 1, -u[i][0] * v[i][1], -u[i][1] * v[i][1], -v[i][1]])

    # TODO: 2.solve H with A
    _, _, Vt=np.linalg.svd(A)
    V = np.transpose(Vt)
    H = V[:, -1]
    H = (H/np.sum(H)).reshape((3,3))
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    X, Y = np.meshgrid(np.arange(xmin, xmax),np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    homogeneous = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), np.ones_like(X.reshape(-1, 1))), axis = 1)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # (3,3) * (N,3)^T = (3,N)
        transformed_coord =  H_inv @ homogeneous.T # np.dot(H_inv, homogeneous.T)
        # inhomogeneous, (N,3)
        transformed_coord = (transformed_coord / transformed_coord[2]).T
        transformed_coord = transformed_coord.astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (transformed_coord[:,0] >= 0) & (transformed_coord[:,0] < w_src) & \
               (transformed_coord[:,1] >= 0) & (transformed_coord[:,1] < h_src)
        # TODO: 5.filter the valid coordinates using previous obtained mask
        valid_dst = homogeneous[mask][:,:-1].T
        valid_src = transformed_coord[mask][:,:-1].T
        # TODO: 6. assign to destination image using advanced array indicing
        dst[valid_dst[1],valid_dst[0]] = src[valid_src[1],valid_src[0]]

        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # (3,3) * (N,3)^T = (3,N)
        transformed_coord = H @ homogeneous.T # np.dot(H, homogeneous.T)
        # inhomogeneous, (N,3)
        transformed_coord = (transformed_coord / transformed_coord[2]).T
        transformed_coord = transformed_coord.astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (transformed_coord[:,0] >= 0) & (transformed_coord[:,0] < w_dst) & \
               (transformed_coord[:,1] >= 0) & (transformed_coord[:,1] < h_dst)
        # TODO: 5.filter the valid coordinates using previous obtained mask
        valid_src = homogeneous[mask][:,:-1].T
        valid_dst = transformed_coord[mask][:,:-1].T
        # TODO: 6. assign to destination image using advanced array indicing
        dst[valid_dst[1],valid_dst[0]] = src[valid_src[1],valid_src[0]]

    return dst 
