"""Feature extraction transformations."""

import logging
import re
from typing import Any

import pandas as pd

from src.core import DataContainer, DataStage

logger = logging.getLogger(__name__)


class PersonalInfoExtractorStage(DataStage):
    """Extract personal information (gender, age, birthday month)."""
    
    def __init__(self):
        """Initialize personal info extractor."""
        super().__init__("PersonalInfoExtractor")
    
    def _parse_gender(self, value: str) -> int:
        """Extract gender: 0 for male, 1 for female."""
        if not isinstance(value, str):
            return 0
        gender_part = value.split(',')[0].strip()
        return 0 if gender_part in ['Мужчина', 'Male'] else 1
    
    def _parse_age(self, value: str) -> int:
        """Extract age in years."""
        if not isinstance(value, str):
            return -1
        parts = value.split(',')
        if len(parts) < 2:
            return -1
        age_str = parts[1].strip().replace('\xa0', ' ').split(' ')[0]
        try:
            return int(age_str)
        except ValueError:
            return -1
    
    def _parse_birth_month(self, value: str) -> int:
        """Extract birth month as integer (0-11)."""
        if not isinstance(value, str):
            return -1
        parts = value.split(',')
        if len(parts) < 3:
            return -1
        
        month_str = parts[2].strip().replace('\xa0', ' ').split(' ')[-2]
        month_map = {
            'January': 0, 'января': 0, 'February': 1, 'февраля': 1,
            'March': 2, 'марта': 2, 'April': 3, 'апреля': 3,
            'May': 4, 'мая': 4, 'June': 5, 'июня': 5,
            'July': 6, 'июля': 6, 'August': 7, 'августа': 7,
            'September': 8, 'сентября': 8, 'October': 9, 'октября': 9,
            'November': 10, 'ноября': 10, 'December': 11, 'декабря': 11,
        }
        return month_map.get(month_str, -1)
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract personal information features."""
        df = container.processed_data.copy()
        col = "Пол, возраст"
        
        df['gender'] = df[col].apply(self._parse_gender)
        df['age'] = df[col].apply(self._parse_age)
        df['birth_month'] = df[col].apply(self._parse_birth_month)
        df = df.drop(columns=[col])
        
        container.processed_data = df
        logger.info("Extracted personal information features")
        return container


class SalaryExtractorStage(DataStage):
    """Extract and normalize salary information."""
    
    CURRENCY_RATES = {
        'руб.': 1.0, 'USD': 73.35, 'RUB': 1.0, 'KZT': 0.18,
        'бел. руб.': 2.28, 'EUR': 85.86, 'грн.': 2.72, 'сум': 0.005,
        'KGS': 0.98, 'UAH': 2.5, 'BYN': 2.5, 'AZN': 41.1, 'som': 0.005,
    }
    
    def __init__(self):
        """Initialize salary extractor."""
        super().__init__("SalaryExtractor")
    
    def _normalize_salary(self, salary_str: str) -> float:
        """Convert salary to rubles."""
        if not isinstance(salary_str, str):
            return 0.0
        
        cleaned = salary_str.replace('\xa0', ' ').strip().split(' ')
        digits = ''
        currency = ''
        
        for idx, token in enumerate(cleaned):
            if token.isdigit():
                digits += token
            else:
                currency = ' '.join(cleaned[idx:])
                break
        
        if not digits:
            return 0.0
        
        try:
            amount = float(digits)
            rate = self.CURRENCY_RATES.get(currency.strip(), 1.0)
            return amount * rate
        except ValueError:
            return 0.0
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract salary feature."""
        df = container.processed_data.copy()
        df['salary_rub'] = df['ЗП'].apply(self._normalize_salary)
        df = df.drop(columns=['ЗП'])
        
        container.processed_data = df
        logger.info("Extracted salary feature")
        return container


class ExperienceExtractorStage(DataStage):
    """Extract work experience in months."""
    
    def __init__(self):
        """Initialize experience extractor."""
        super().__init__("ExperienceExtractor")
    
    def _extract_months(self, exp_str: str) -> int:
        """Parse experience string to total months."""
        if not isinstance(exp_str, str):
            return 0
        
        years_match = re.search(r'(\d+)\s*(?:год|года|лет)', exp_str)
        months_match = re.search(r'(\d+)\s*(?:месяц|месяца|месяцев)', exp_str)
        
        total = 0
        if years_match:
            total += int(years_match.group(1)) * 12
        if months_match:
            total += int(months_match.group(1))
        
        return total
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract experience feature."""
        df = container.processed_data.copy()
        exp_col = 'Опыт (двойное нажатие для полной версии)'
        df['experience_months'] = df[exp_col].apply(self._extract_months)
        df = df.drop(columns=[exp_col])
        
        container.processed_data = df
        logger.info("Extracted experience feature")
        return container


class LocationExtractorStage(DataStage):
    """Extract and categorize location information."""
    
    REGION_MAPPING = {
        "Moscow & Oblast": [
            "Москва", "Moscow", "Зеленоград", "Подольск", "Балашиха", "Химки", "Мытищи",
            "Королев", "Люберцы", "Красногорск", "Одинцово", "Домодедово", "Щелково",
            "Серпухов", "Раменское", "Долгопрудный", "Реутов", "Пушкино", "Лобня"
        ],
        "Saint Petersburg & Oblast": [
            "Санкт-Петербург", "Saint Petersburg", "Гатчина", "Выборг", "Всеволожск",
            "Сосновый Бор", "Кириши", "Тихвин", "Сертолово"
        ],
        "Central Federal District": [
            "Воронеж", "Ярославль", "Рязань", "Тверь", "Тула", "Липецк", "Курск",
            "Брянск", "Иваново", "Белгород", "Владимир", "Калуга", "Орел", "Смоленск",
            "Тамбов", "Кострома", "Старый Оскол"
        ],
        "Volga Federal District": [
            "Казань", "Kazan", "Нижний Новгород", "Самара", "Уфа", "Пермь", "Саратов",
            "Тольятти", "Ижевск", "Ульяновск", "Оренбург", "Пенза", "Набережные Челны",
            "Чебоксары", "Киров", "Саранск", "Стерлитамак", "Йошкар-Ола"
        ],
        "South and North Caucasus Federal District": [
            "Краснодар", "Ростов-на-Дону", "Волгоград", "Сочи", "Ставрополь", "Астрахань",
            "Севастополь", "Симферополь", "Новороссийск", "Таганрог", "Махачкала",
            "Владикавказ", "Грозный", "Майкоп", "Пятигорск"
        ],
        "Ural Federal District": [
            "Екатеринбург", "Yekaterinburg", "Челябинск", "Тюмень", "Магнитогорск",
            "Сургут", "Нижневартовск", "Курган", "Новый Уренгой", "Ноябрьск", "Ханты-Мансийск"
        ],
        "Siberian Federal District": [
            "Новосибирск", "Novosibirsk", "Красноярск", "Омск", "Томск", "Барнаул",
            "Иркутск", "Кемерово", "Новокузнецк", "Абакан", "Братск", "Ангарск"
        ],
        "Far Eastern Federal District": [
            "Владивосток", "Хабаровск", "Улан-Удэ", "Чита", "Благовещенск", "Якутск",
            "Петропавловск-Камчатский", "Южно-Сахалинск", "Находка"
        ],
        "Kazakhstan": [
            "Алматы", "Almaty", "Нур-Султан", "Астана", "Astana", "Шымкент", "Актобе",
            "Караганда", "Атырау", "Актау", "Павлодар", "Уральск"
        ],
        "Belarus": [
            "Минск", "Minsk", "Гомель", "Витебск", "Могилев", "Гродно", "Брест"
        ],
        "Other countries / CIS": [
            "Киев", "Kyiv", "Ташкент", "Бишкек", "Тбилиси", "Баку", "Ереван", "Рига", "Вильнюс"
        ]
    }
    
    def __init__(self):
        """Initialize location extractor."""
        super().__init__("LocationExtractor")
    
    def _categorize_city(self, city_str: str) -> str:
        """Categorize city into region."""
        if not isinstance(city_str, str):
            return "Other"
        
        city_name = city_str.split(',')[0].strip()
        for region, cities in self.REGION_MAPPING.items():
            if city_name in cities:
                return region
        return "Other"
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract location feature."""
        df = container.processed_data.copy()
        df['region'] = df['Город'].apply(self._categorize_city)
        df = df.drop(columns=['Город'])
        
        container.processed_data = df
        logger.info("Extracted location feature")
        return container


class EmploymentExtractorStage(DataStage):
    """Extract employment type features."""
    
    EMPLOYMENT_TYPES = {
        "full_time": ["полная занятость", "full time"],
        "part_time": ["частичная занятость", "part time"],
        "project": ["проектная работа", "project work"],
        "internship": ["стажировка", "work placement"],
        "volunteering": ["волонтерство", "volunteering"]
    }
    
    def __init__(self):
        """Initialize employment extractor."""
        super().__init__("EmploymentExtractor")
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract employment type features."""
        df = container.processed_data.copy()
        emp_col = "Занятость"
        
        for emp_type, keywords in self.EMPLOYMENT_TYPES.items():
            df[f"employment_{emp_type}"] = df[emp_col].apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        df = df.drop(columns=[emp_col])
        container.processed_data = df
        logger.info("Extracted employment type features")
        return container


class ScheduleExtractorStage(DataStage):
    """Extract work schedule features."""
    
    SCHEDULE_TYPES = {
        "full_day": ["полный день", "full day"],
        "flexible": ["гибкий график", "flexible schedule"],
        "shift": ["сменный график", "shift schedule"],
        "remote": ["удаленная работа", "remote working"],
        "rotation": ["вахтовый метод", "rotation based work"]
    }
    
    def __init__(self):
        """Initialize schedule extractor."""
        super().__init__("ScheduleExtractor")
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract schedule features."""
        df = container.processed_data.copy()
        schedule_col = "График"
        
        for sched_type, keywords in self.SCHEDULE_TYPES.items():
            df[f"schedule_{sched_type}"] = df[schedule_col].apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        df = df.drop(columns=[schedule_col])
        container.processed_data = df
        logger.info("Extracted schedule features")
        return container


class EducationExtractorStage(DataStage):
    """Extract education level features."""
    
    EDUCATION_LEVELS = {
        "incomplete_higher": ["неоконченное высшее", "incomplete higher"],
        "higher": ["высшее образование", "higher education"],
        "secondary_special": ["среднее специальное", "secondary special"],
        "secondary": ["среднее образование", "secondary education"]
    }
    
    def __init__(self):
        """Initialize education extractor."""
        super().__init__("EducationExtractor")
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract education features."""
        df = container.processed_data.copy()
        edu_col = "Образование и ВУЗ"
        
        for edu_level, keywords in self.EDUCATION_LEVELS.items():
            df[f"education_{edu_level}"] = df[edu_col].apply(
                lambda x: 1 if any(kw in str(x).lower() for kw in keywords) else 0
            )
        
        df = df.drop(columns=[edu_col])
        container.processed_data = df
        logger.info("Extracted education features")
        return container


class MiscExtractorStage(DataStage):
    """Extract miscellaneous features (resume date, car ownership)."""
    
    def __init__(self):
        """Initialize misc extractor."""
        super().__init__("MiscExtractor")
    
    def _is_old_resume(self, date_str: str) -> int:
        """Check if resume is old (before 2019)."""
        if not isinstance(date_str, str):
            return 0
        try:
            year = int(date_str.split('.')[2].split(' ')[0])
            return 1 if year <= 2018 else 0
        except (IndexError, ValueError):
            return 0
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Extract miscellaneous features."""
        df = container.processed_data.copy()
        
        # Resume age
        if "Обновление резюме" in df.columns:
            df['resume_old'] = df["Обновление резюме"].apply(self._is_old_resume)
            df = df.drop(columns=["Обновление резюме"])
        
        # Car ownership
        if "Авто" in df.columns:
            df['has_car'] = df["Авто"].apply(
                lambda x: 1 if x == 'Имеется собственный автомобиль' else 0
            )
            df = df.drop(columns=["Авто"])
        
        # Drop last job position (used for labeling, prevent leakage)
        if "Последенее/нынешнее место работы" in df.columns:
            df = df.drop(columns=["Последенее/нынешнее место работы"])
        if "Последеняя/нынешняя должность" in df.columns:
            df = df.drop(columns=["Последеняя/нынешняя должность"])
        
        container.processed_data = df
        logger.info("Extracted miscellaneous features")
        return container

