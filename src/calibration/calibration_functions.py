# Obtener los puntos del tablero de ajedrez
def get_chessboard_points(chessboard_shape, dx, dy):
    """Devuelve los puntos del tablero de ajedrez en el sistema de coordenadas del mundo."""
    objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
    objp *= np.array([dx, dy], dtype=np.float32)
    return objp