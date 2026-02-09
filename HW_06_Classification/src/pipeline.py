from src.core import Handler
from src.handlers.io import LoadCSVHandler, SaveDataHandler
from src.handlers.parsing import (
    ParseGenderAgeBirthdayHandler, ParseSalaryHandler, ParseJobHandler,
    ParseCityHandler, ParseEmploymentHandler, ParseWorkScheduleHandler,
    ParseExperienceHandler, ParseLastPlaceHandler, ParseLastJobHandler,
    ParseEducationHandler, ParseResumeHandler, ParseAutoHandler
)
from src.handlers.preprocessing import EncodeCategoricalFeaturesHandler, SplitDataHandler

def build_pipeline() -> Handler:
    """
    Builds the full data processing pipeline by chaining together all handlers in the required order.

    Returns:
        Handler: The first handler in the pipeline (LoadCSVHandler).
    """
    load = LoadCSVHandler()
    gender_age = ParseGenderAgeBirthdayHandler()
    salary = ParseSalaryHandler()
    job = ParseJobHandler()
    city = ParseCityHandler()
    employment = ParseEmploymentHandler()
    work_schedule = ParseWorkScheduleHandler()
    experience = ParseExperienceHandler()
    last_place = ParseLastPlaceHandler()
    last_job = ParseLastJobHandler()
    education = ParseEducationHandler()
    resume = ParseResumeHandler()
    auto = ParseAutoHandler()
    encode_categorical_features = EncodeCategoricalFeaturesHandler()

    split_data = SplitDataHandler()

    save_data = SaveDataHandler()

    load.set_next(gender_age)\
        .set_next(salary)\
        .set_next(job)\
        .set_next(city)\
        .set_next(employment)\
        .set_next(work_schedule)\
        .set_next(experience)\
        .set_next(last_place)\
        .set_next(last_job)\
        .set_next(education)\
        .set_next(resume)\
        .set_next(auto)\
        .set_next(encode_categorical_features)\
        .set_next(split_data)\
        .set_next(save_data)
    
    return load
