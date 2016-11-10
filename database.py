import dill
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative.api import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.sql.schema import Column, ForeignKey
from sqlalchemy.sql.sqltypes import DateTime, Float, Integer, String, PickleType

Base = declarative_base()


class Scenario(Base):
    __tablename__ = 'scenarios'

    id = Column(Integer, primary_key=True)
    description = Column(String)
    generations = Column(Integer)
    population_size = Column(Integer)


class Individual(Base):
    __tablename__ = 'individuals'

    scenario_id = Column(Integer, ForeignKey('scenarios.id'), primary_key=True)
    individual_number = Column(Integer, primary_key=True)
    generation = Column(Integer, primary_key=True)
    genotype = Column(PickleType(pickler=dill))
    fitness = Column(Float)
    timestamp = Column(DateTime)


class Db:
    def __init__(self, path, echo=True):
        self.engine = create_engine(path, echo=echo)
        Base.metadata.create_all(self.engine)
        from sqlalchemy.orm import scoped_session
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def save_scenario(self, scenario, session=None):
        if session is None:
            session = self.Session()

        session.add(scenario)
        session.commit()

        return scenario

    def save_individual(self, individual, session=None):
        if session is None:
            session = self.Session()

        session.add(individual)

        session.commit()

        return individual

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


def get_db(path):
    return Db(path, echo=False)
