from neat.species import Species
from uuid import UUID

Species.__repr__ = lambda self: f'Species {UUID(int=self.ID).hex}'
