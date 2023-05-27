import numpy as np
import scipy.signal as signal

def _not(func):
    def not_func(x):
        return not func(x)
    return not_func

class Board(object):
    _mine_threshold = 0
    def __init__(self, value, mask):
        self._value = value
        self._mask = mask
        self._pad_width = [{0: (1,0), l-1: (0,1)} for l in value.shape]

    def update_view(self, view, point):
        return 'game over' if self._is_mine(point) else view.open(self, [point])

    def at(self, point):
        return self._value[point]

    def neighbours_of(self, point):
        return self._all_neighbours(point) if self._safe(point) else []  # stop DFS

    def _is_mine(self, point):
        return self._value[point] < self._mine_threshold

    def _safe(self, point):
        return self._value[point] == self._mine_threshold

    def _all_neighbours(self, point):
        return [tuple (x-1 + point) for x in np.argwhere((self.pad(point)==0) * self._mask) ]

    def adjacent_to(self, point):
        return self._value[[slice(max(0, p-1), min(self._value.shape[i], p+2)) for i, p in enumerate(point)]]

    def pad(self, point):
        return np.pad(self.adjacent_to(point), self.pad_width(point), 'constant', constant_values = 1  )

    def pad_width(self, point):
        return [self._pad_width[i].get(p, (0,0)) for i, p in enumerate(point)]

    @classmethod
    def create(cls, shape, number_of_mines, mask):
        return cls(signal.convolve2d(_new_board(shape, number_of_mines), mask, mode='same'), mask==-1)

def _seed_mines(flatten_board, number_of_mines, fill=-1):
    flatten_board[np.random.choice(list(range(flatten_board.shape[0])), replace=False, size=number_of_mines)] = fill
    return flatten_board

def _new_board(shape, number_of_mines):
    return _seed_mines(_flatten_board(shape), number_of_mines).reshape(shape)

def _flatten_board(shape):
    return np.zeros(shape).flatten()

class View(object):
    visited_threshold = -1
    def __init__(self, value):
        self._value = value

    def open(self, board, points):
        self._visit_all(self._not_visited(points), board)
        return self

    def _visit_all(self, points, board):
        for point in points: self._visit(point, board)

    def _visit(self, point, board):
        self._value[point] = board.at(point) # DFS
        self.open(board, board.neighbours_of(point))

    def _not_visited(self, points):
        return filter(_not(self._visited), points)

    def _visited(self, point):
        return self._value[point] > self.visited_threshold

    def __repr__(self):
        return self._value.__repr__()

    @classmethod
    def create(cls, shape):
        return cls(np.full(shape, cls.visited_threshold))

def default_mask():
    return np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])

class Minesweaper(object):
    def __init__(self, board, view):
        self._board = board # ground truth
        self._view = view # user screen

    def command(self, *point):
        self._view = self._board.update_view(self._view, point)

    @classmethod
    def create(cls, boundaries=(8,8), number_of_mines=10, mask=default_mask()):
        return cls(Board.create(boundaries, number_of_mines, mask), View.create(boundaries))

    def __repr__(self):
        return self._view.__repr__()

game = Minesweaper.create()
print (game)
game.command(1,1)
print (game)
