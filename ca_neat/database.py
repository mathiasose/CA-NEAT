import uuid

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.sql.functions import now
from sqlalchemy.sql.schema import Column, ForeignKey, ForeignKeyConstraint
from sqlalchemy.sql.sqltypes import DateTime, Float, Integer, LargeBinary, String
from sqlalchemy.types import CHAR, TypeDecorator
from sqlalchemy_utils.functions.database import create_database, database_exists


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    """
    impl = CHAR

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                # hexstring
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)


Base = declarative_base()


class Scenario(Base):
    __tablename__ = 'scenarios'

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    generations = Column(Integer)
    population_size = Column(Integer)

    def __repr__(self):
        return 'Scenario {}'.format(self.id)


class Individual(Base):
    __tablename__ = 'individuals'

    scenario_id = Column(Integer, ForeignKey('scenarios.id'), primary_key=True, index=True)
    individual_number = Column(Integer, primary_key=True, index=True)
    generation = Column(Integer, primary_key=True, index=True)
    genotype = Column(LargeBinary)
    fitness = Column(Float, index=True)
    λ = Column(Float)
    species = Column(GUID, index=True)
    timestamp = Column(DateTime, default=now(), index=True)

    def __repr__(self):
        return 's={}, g={}, n={}, f={}, λ={}'.format(
            self.scenario_id,
            self.generation,
            self.individual_number,
            round(self.fitness, 2),
            'N/A' if self.λ is None else round(self.λ, 2),
        )


class Innovation(Base):
    __tablename__ = 'innovations'

    id = Column(Integer, primary_key=True)
    scenario_id = Column(Integer)
    generation = Column(Integer)
    individual_number = Column(Integer)

    __table_args__ = (
        ForeignKeyConstraint(
            [scenario_id, generation, individual_number],
            [Individual.scenario_id, Individual.generation, Individual.individual_number]
        ),
        {}
    )


class Db:
    def __init__(self, path, echo=True):
        self.engine = create_engine(path, echo=echo)
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        Base.metadata.create_all(self.engine)
        from sqlalchemy.orm import scoped_session
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def save_scenario(self, scenario, session=None):
        if session is None:
            session = self.Session()

        session.add(scenario)
        session.commit()

        return scenario

    def get_scenarios(self, session=None):
        if session is None:
            session = self.Session()

        return session.query(Scenario)

    def get_scenario(self, scenario_id, session=None):
        if session is None:
            session = self.Session()

        return session.query(Scenario).get(scenario_id)

    def get_individuals(self, scenario_id, session=None):
        if session is None:
            session = self.Session()

        return session.query(Individual) \
            .filter(Individual.scenario_id == scenario_id)

    def get_generation(self, scenario_id, generation, session=None):
        if session is None:
            session = self.Session()

        return self.get_individuals(scenario_id, session) \
            .filter(Individual.generation == generation)

    def get_species_birth_generation(self, scenario_id, species_id, session=None):
        if session is None:
            session = self.Session()

        return self.get_individuals(scenario_id=scenario_id, session=session) \
            .with_entities(Individual.generation) \
            .filter(Individual.species == uuid.UUID(int=species_id)) \
            .order_by(Individual.generation) \
            .first() \
            .generation

    def get_innovations(self, scenario_id, session=None):
        if session is None:
            session = self.Session()

        return session.query(Individual).join(Innovation) \
            .filter(Innovation.scenario_id == scenario_id) \
            .filter(Innovation.scenario_id == Individual.scenario_id) \
            .filter(Innovation.generation == Individual.generation) \
            .filter(Innovation.individual_number == Individual.individual_number) \
            .values(Individual.genotype)


def get_db(path):
    return Db(path, echo=False)
