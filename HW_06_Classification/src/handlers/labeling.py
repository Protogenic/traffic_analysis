import logging

from src.core import Handler, PipelineContext


class LabelGradeHandler(Handler):
    """
    Generates target variable 'grade' (Junior/Middle/Senior) using title and experience,
    then drops these features to prevent target leakage.
    """

    _JUNIOR_KW = ['junior', 'jun ', 'jr ', 'trainee', 'стажер', 'стажёр', 'intern', 'младший', 'начинающий']
    _SENIOR_KW = ['senior', 'sr ', 'lead', 'principal', 'staff', 'ведущий', 'главный', 'руководитель', 'team lead', 'architect', 'head of', 'expert']
    _MIDDLE_KW = ['middle', 'mid ', 'мидл', 'мидлл']

    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Generate 'grade' target and remove source features.

        Args:
            ctx: Pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context updated with 'grade' target and clean feature set.
        """
        logging.info("LabelGradeHandler: Starting to label grades")
        df = ctx.df.copy()

        raw_title_col = 'Ищет работу на должность:'
        if raw_title_col in df.columns:
            df['_title'] = df[raw_title_col].fillna('').str.lower()
        elif 'job' in df.columns:
            df['_title'] = df['job'].fillna('').str.lower()
        else:
            df['_title'] = ''

        def _label(row) -> str:
            """Determine grade based on title keywords and experience."""
            title: str = row['_title']
            exp: int = row.get('experience_months', 0)

            is_junior = any(kw in title for kw in self._JUNIOR_KW)
            is_senior = any(kw in title for kw in self._SENIOR_KW)
            is_middle = any(kw in title for kw in self._MIDDLE_KW)

            if is_senior and is_junior: return 'Senior' if exp > 36 else 'Junior'
            if is_senior: return 'Senior'
            if is_junior: return 'Middle' if exp > 60 else 'Junior'
            if is_middle: return 'Senior' if exp > 96 else 'Middle'

            if exp <= 18: return 'Junior'
            if exp <= 60: return 'Middle'
            return 'Senior'

        df['grade'] = df.apply(_label, axis=1)

        # Drop features used for labeling to prevent leakage
        cols_to_drop = ['_title']
        if raw_title_col in df.columns: cols_to_drop.append(raw_title_col)
        if 'experience_months' in df.columns: cols_to_drop.append('experience_months')
        if 'Последеняя/нынешняя должность' in df.columns: cols_to_drop.append('Последеняя/нынешняя должность')

        df = df.drop(columns=cols_to_drop)

        ctx.df = df
        ctx.target = df['grade'].values

        logging.info(f"LabelGradeHandler: Distribution:\n{df['grade'].value_counts().to_string()}")
        logging.info(f"LabelGradeHandler: Dropped {cols_to_drop} to prevent leakage.")
        return ctx
