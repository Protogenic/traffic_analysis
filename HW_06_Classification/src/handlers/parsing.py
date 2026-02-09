import logging
import re

import pandas as pd

from src.core import Handler, PipelineContext

class ParseGenderAgeBirthdayHandler(Handler):
    """Handler for parsing gender, age, and birthday information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Пол, возраст' column into separate gender, age, and birthday_month columns.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with parsed features.
        """
        logging.info(f"ParseGenderAgeBirthdayHandler: Starting to parse gender, age and birthday month")
        df = ctx.df.copy()
        
        male_values = ['Мужчина', 'Male']
        
        def extract_gender(value: str) -> str:
            raw_gender = value.split(',')[0].strip()
            if raw_gender in male_values:
                return 0 # Male
            return 1 # Female

        def extract_age(value: str) -> str:
            data = value.split(',')
            if len(data) < 2:
                return -1
            raw_age = data[1].strip().replace('\xa0', ' ')
            raw_age = raw_age.split(' ')[0]
            return int(raw_age)

        def extract_birthday_month(value: str) -> str:
            data = value.split(',')
            if len(data) < 3:
                return -1
            raw_birthday_month = data[2].strip().replace('\xa0', ' ')
            raw_birthday_month = raw_birthday_month.split(' ')[-2]
            match raw_birthday_month:
                case 'January' | 'января': return 0
                case 'February' | 'февраля': return 1
                case 'March' | 'марта': return 2
                case 'April' | 'апреля': return 3
                case 'May' | 'мая': return 4
                case 'June' | 'июня': return 5
                case 'July' | 'июля': return 6
                case 'August' | 'августа': return 7
                case 'September' | 'сентября': return 8    
                case 'October' | 'октября': return 9
                case 'November' | 'ноября': return 10
                case 'December' | 'декабря': return 11
                case _: return -1

        df["gender"] = df["Пол, возраст"].apply(extract_gender)
        df["age"] = df["Пол, возраст"].apply(extract_age)
        df["birthday_month"] = df["Пол, возраст"].apply(extract_birthday_month)

        df = df.drop(columns=["Пол, возраст"])

        ctx.df = df
        logging.info(f"ParseGenderAgeHandler: Parsed gender, age and birthday month")
        return ctx

class ParseSalaryHandler(Handler):
    """Handler for parsing salary information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'ЗП' column and convert all salaries to rubles.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'salary_rub' column.
        """
        logging.info(f"ParseSalaryHandler: Starting to parse salary")
        df = ctx.df.copy()

        currency_rates = {
            'руб.': 1.0, 'USD': 73.35, 'RUB': 1.0, 'KZT': 0.18,
            'бел. руб.': 2.28, 'EUR': 85.86, 'грн.': 2.72, 'сум': 0.005,
            'KGS': 0.98, 'UAH': 2.5, 'BYN': 2.5, 'AZN': 41.1, 'som': 0.005,
        }
        
        def extract_salary(value: str) -> str:
            value = value.replace('\xa0', ' ').strip().split(' ')
            number = ''
            currency = ''

            for idx, cur in enumerate(value):
                if cur.isdigit():
                    number += cur
                else:
                    currency = ' '.join(value[idx:])
                    break
            return currency_rates[currency.strip()] * float(number)

        df["salary_rub"] = df["ЗП"].apply(extract_salary)
        df = df.drop(columns=["ЗП"])

        ctx.df = df
        logging.info(f"ParseSalaryHandler: Parsed salary")
        return ctx

class ParseJobHandler(Handler):
    """Handler for parsing job information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Normalize job titles to the top 133 most common ones, grouping others as 'other'.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with normalized 'job' column.
        """
        logging.info(f"ParseJobHandler: Starting to parse job")
        df = ctx.df.copy()

        job_count = df['Ищет работу на должность:'].value_counts()[:133]

        def extract_job(value: str) -> str:
            if value in job_count:
                return value
            return "other"

        df["job"] = df["Ищет работу на должность:"].apply(extract_job)
        df = df.drop(columns=["Ищет работу на должность:"])

        ctx.df = df
        logging.info(f"ParseJobHandler: Parsed job")
        return ctx

class ParseCityHandler(Handler):
    """Handler for parsing city information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Group cities into regions/federal districts.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'city' region column.
        """
        logging.info(f"ParseCityHandler: Starting to parse city")
        df = ctx.df.copy()

        regions_map = {
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

        def extract_city(value: str) -> str:
            city_name = value.split(',')[0].strip()
            for region, cities in regions_map.items():
                if city_name in cities:
                    return region
            return "Other"

        df["city"] = df["Город"].apply(extract_city)
        df = df.drop(columns=["Город"])

        ctx.df = df
        logging.info(f"ParseCityHandler: Parsed city")
        return ctx

class ParseEmploymentHandler(Handler):
    """Handler for parsing employment information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Занятость' column into binary flags.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with employment flags.
        """
        logging.info(f"ParseEmploymentHandler: Starting to parse employment")
        df = ctx.df.copy()

        employment_map = {
            "full_time": ["полная занятость", "full time"],
            "part_time": ["частичная занятость", "part time"],
            "project": ["проектная работа", "project work"],
            "internship": ["стажировка", "work placement"],
            "volunteering": ["волонтерство", "volunteering"]
        }

        for column_name, keywords in employment_map.items():
            def check_employment(value: str) -> int:
                value_lower = value.lower()
                if any(keyword in value_lower for keyword in keywords):
                    return 1
                return 0
            
            df[f"emp_{column_name}"] = df["Занятость"].apply(check_employment)

        df = df.drop(columns=["Занятость"])

        ctx.df = df
        logging.info(f"ParseEmploymentHandler: Parsed employment")
        return ctx

class ParseWorkScheduleHandler(Handler):
    """Handler for parsing work schedule information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'График' column into binary flags.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with schedule flags.
        """
        logging.info(f"ParseWorkScheduleHandler: Starting to parse work schedule")
        df = ctx.df.copy()

        schedule_map = {
            "full_day": ["полный день", "full day"],
            "flexible": ["гибкий график", "flexible schedule"],
            "shift": ["сменный график", "shift schedule"],
            "remote": ["удаленная работа", "remote working"],
            "rotation": ["вахтовый метод", "rotation based work"]
        }

        for column_name, keywords in schedule_map.items():
            def check_schedule(value: str) -> int:
                value_lower = value.lower()
                if any(keyword in value_lower for keyword in keywords):
                    return 1
                return 0
            
            df[f"sch_{column_name}"] = df["График"].apply(check_schedule)

        df = df.drop(columns=["График"])

        ctx.df = df
        logging.info(f"ParseWorkScheduleHandler: Parsed work schedule")
        return ctx

class ParseExperienceHandler(Handler):
    """Handler for parsing experience information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Опыт' string into total months.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'experience_months' column.
        """
        logging.info("ParseExperienceHandler: Parsing experience")
        df = ctx.df.copy()

        def extract(value: str) -> int:
            if not isinstance(value, str): 
                return 0
            years_match = re.search(r'(\d+)\s*(?:год|года|лет)', value)
            months_match = re.search(r'(\d+)\s*(?:месяц|месяца|месяцев)', value)
            total = 0
            if years_match:
                total += int(years_match.group(1)) * 12
            if months_match:
                total += int(months_match.group(1))
            return total
        
        df["experience_months"] = df["Опыт (двойное нажатие для полной версии)"].apply(extract)
        ctx.df = df.drop(columns=["Опыт (двойное нажатие для полной версии)"])
        logging.info("ParseExperienceHandler: Done")
        return ctx

class ParseLastPlaceHandler(Handler):
    """Handler for parsing last place information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Drop 'Последенее/нынешнее место работы' column.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context with the column removed.
        """
        logging.info("ParseLastPlaceHandler: Dropping column")
        ctx.df = ctx.df.drop(columns=["Последенее/нынешнее место работы"])
        return ctx

class ParseLastJobHandler(Handler):
    """Handler for parsing last job information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Normalize 'Последеняя/нынешняя должность' to top 133 or 'other'.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'last_job' column.
        """
        logging.info("ParseLastJobHandler: Parsing last job")
        df = ctx.df.copy()
        
        # Note: ideally reuse top_jobs from ParseJobHandler
        jobs = df['job'].value_counts().index if 'job' in df else []
        
        df["last_job"] = df["Последеняя/нынешняя должность"].apply(
            lambda job_title: job_title if job_title in jobs else "other"
        )
        ctx.df = df.drop(columns=["Последеняя/нынешняя должность"])
        logging.info("ParseLastJobHandler: Done")
        return ctx

class ParseEducationHandler(Handler):
    """Handler for parsing education information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Образование и ВУЗ' into binary education level flags.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with education flags.
        """
        logging.info("ParseEducationHandler: Parsing education")
        df = ctx.df.copy()
        
        mapping = {
            "incomplete_higher": ["неоконченное высшее", "incomplete higher"],
            "higher": ["высшее образование", "higher education"],
            "secondary_special": ["среднее специальное", "secondary special"],
            "secondary": ["среднее образование", "secondary education"]
        }

        for column_suffix, keywords in mapping.items():
            df[f"edu_{column_suffix}"] = df["Образование и ВУЗ"].apply(
                lambda education_string: 1 if any(keyword in education_string.lower() for keyword in keywords) else 0
            )

        ctx.df = df.drop(columns=["Образование и ВУЗ"])
        logging.info("ParseEducationHandler: Done")
        return ctx

class ParseResumeHandler(Handler):
    """Handler for parsing resume information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Обновление резюме' to create 'old_resume' flag.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'old_resume' flag.
        """
        logging.info("ParseResumeHandler: Parsing resume date")
        df = ctx.df.copy()

        def is_old(date_string: str) -> int:
            try:
                year = int(date_string.split('.')[2].split(' ')[0])
                return 0 if year > 2018 else 1
            except: return 0

        df["old_resume"] = df["Обновление резюме"].apply(is_old)
        ctx.df = df.drop(columns=["Обновление резюме"])
        logging.info("ParseResumeHandler: Done")
        return ctx

class ParseAutoHandler(Handler):
    """Handler for parsing auto information."""
    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Parse 'Авто' to create 'auto' ownership flag.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'auto' flag.
        """
        logging.info("ParseAutoHandler: Parsing auto")
        df = ctx.df.copy()
        df["auto"] = df["Авто"].apply(
            lambda auto_status: 1 if auto_status == 'Имеется собственный автомобиль' else 0
        )
        ctx.df = df.drop(columns=["Авто"])
        logging.info("ParseAutoHandler: Done")
        return ctx
