import random, math
from _functools import reduce
from copy import copy
from builtins import isinstance
from resource import setrlimit, RLIMIT_AS, RLIMIT_DATA


class NPuzzle:
	"""
	Reprezentarea unei stări a problemei și a istoriei mutărilor care au adus starea aici.
	
	Conține funcționalitate pentru
	- afișare
	- citirea unei stări dintr-o intrare pe o linie de text
	- obținerea sau ștergerea istoriei de mutări
	- obținerea variantei rezolvate a acestei probleme
	- verificarea dacă o listă de mutări fac ca această stare să devină rezolvată.
	"""

	NMOVES = 4
	UP, DOWN, LEFT, RIGHT = range(NMOVES)
	ACTIONS = [UP, DOWN, LEFT, RIGHT]
	names = "UP, DOWN, LEFT, RIGHT".split(", ")
	BLANK = ' '
	delta = dict(zip(ACTIONS, [(-1, 0), (1, 0), (0, -1), (0, 1)]))
	
	PAD = 2
	
	def __init__(self, puzzle : list[int | str], movesList : list[int] = []):
		"""
		Creează o stare nouă pe baza unei liste liniare de piese, care se copiază.
		
		Opțional, se poate copia și lista de mutări dată.
		"""
		self.N = len(puzzle)
		self.side = int(math.sqrt(self.N))
		self.r = copy(puzzle)
		self.moves = copy(movesList)
	
	def display(self, show = True) -> str:
		l = "-" * ((NPuzzle.PAD + 1) * self.side + 1)
		aslist = self.r
		
		slices = [aslist[ slice * self.side : (slice+1) * self.side ]  for slice in range(self.side)]
		s = ' |\n| '.join([' '.join([str(e).rjust(NPuzzle.PAD, ' ') for e in line]) for line in slices]) 
	
		s = ' ' + l + '\n| ' + s + ' |\n ' + l
		if show: print(s)
		return s
	def display_moves(self):
		print([names[a] if a is not None else None for a in moves])
		
	def print_line(self):
		return str(self.r)
	
	@staticmethod
	def read_from_line(line : str):
		list = line.strip('\n][').split(', ')
		numeric = [NPuzzle.BLANK if e == "' '" else int(e) for e in list]
		return NPuzzle(numeric)
	
	def clear_moves(self):
		"""Șterge istoria mutărilor pentru această stare."""
		self.moves.clear()
	
	def apply_move_inplace(self, move : int):
		"""Aplică o mutare, modificând această stare."""
		blankpos = self.r.index(NPuzzle.BLANK)
		y, x = blankpos // self.side, blankpos % self.side
		ny, nx = y + NPuzzle.delta[move][0], x + NPuzzle.delta[move][1]
		if ny < 0 or ny >= self.side or nx < 0 or nx >= self.side: return None
		newpos = ny * self.side + nx
		piece = self.r[newpos]
		self.r[blankpos] = piece
		self.r[newpos] = NPuzzle.BLANK
		self.moves.append(move)
		return self
	
	def apply_move(self, move : int):
		"""Construiește o nouă stare, rezultată în urma aplicării mutării date."""
		return self.clone().apply_move_inplace(move)

	def solved(self):
		"""Întoarce varianta rezolvată a unei probleme de aceeași dimensiune."""
		return NPuzzle(list(range(self.N))[1:] + [NPuzzle.BLANK])

	def verify_solved(self, moves : list[int]) -> bool:
		""""Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
		return reduce(lambda s, m: s.apply_move_inplace(m), moves, self.clone()) == self.solved()

	def clone(self):
		return NPuzzle(self.r, self.moves)
	def __str__(self) -> str:
		return str(self.N-1) + "-puzzle:" + str(self.r)
	def __repr__(self) -> str: return str(self)
	def __eq__(self, other):
		return self.r == other.r
	def __lt__(self, other):
		return True
	def __hash__(self):
		return hash(tuple(self.r))
	


MLIMIT = 3 * 10 ** 9 # 2 GB RAM limit
setrlimit(RLIMIT_DATA, (MLIMIT, MLIMIT))

f = open("files/problems4.txt", "r")
input = f.readlines()
f.close()
problems = [NPuzzle.read_from_line(line) for line in input]
problems[0].display()

