import logging
import re

from src.core import Handler, PipelineContext


class FilterITRolesHandler(Handler):
    """
    Filters dataset to include only IT-related roles using two-tier keyword matching.
    """

    # Unambiguous IT roles
    _STRONG_KEYWORDS: list[str] = [
        'разработчик', 'developer', 'программист', 'programmer',
        'devops', 'sre', 'backend', 'frontend', 'fullstack', 'full-stack',
        'full stack', 'data scientist', 'data analyst', 'data engineer',
        'machine learning', 'ml engineer', 'deep learning',
        'qa', 'тестировщик', 'tester', 'test engineer',
        'системный администратор', 'system administrator', 'sysadmin',
        'сетевой инженер', 'network engineer',
        'веб-разработчик', 'web developer',
        'ios', 'android', 'мобильный разработчик', 'mobile developer',
        '1с', '1c',
        'dba', 'database administrator', 'администратор баз данных',
        'информационная безопасность', 'information security',
        'аналитик данных', 'бизнес-аналитик',
        'scrum', 'agile', 'product owner', 'project manager',
        'ui/ux', 'ux', 'ui designer',
        'верстальщик', 'html',
        'техническая поддержка', 'technical support', 'helpdesk',
    ]

    # Tech stack keywords (implies IT role)
    _TECH_KEYWORDS: list[str] = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
        'golang', 'go ', 'rust', 'ruby', 'php', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'django', 'flask', 'spring',
        'docker', 'kubernetes', 'k8s', 'terraform', 'ansible',
        'aws', 'azure', 'gcp', 'sql', 'nosql', 'mongodb', 'postgresql',
        'linux', 'unix',
        'api', 'microservice', 'микросервис',
    ]

    # Weak keywords requiring context
    _WEAK_KEYWORDS: list[str] = [
        'инженер', 'engineer', 'администратор', 'admin',
        'аналитик', 'analyst', 'архитектор', 'architect',
        'консультант', 'consultant', 'менеджер проект', 'project manager',
    ]

    # Context required for weak keywords
    _IT_CONTEXT: list[str] = [
        'по', 'программ', 'софт', 'soft', 'it', 'ит',
        'информац', 'автоматиз', 'асу', 'erp', 'crm', 'sap',
        'devops', 'cloud', 'облач', 'сет', 'network', 'систем',
        'данн', 'data', 'баз', 'database', 'cyber', 'кибер',
        'техн', 'tech', 'цифр', 'digital',
    ]

    def _process(self, ctx: PipelineContext) -> PipelineContext:
        """
        Filter rows where the job title matches IT-related keywords.

        Args:
            ctx: The data pipeline context containing the dataframe.

        Returns:
            PipelineContext: Context with filtered dataframe.
        """
        logging.info("FilterITRolesHandler: Filtering for IT roles")
        df = ctx.df.copy()
        initial_count = len(df)
        col = 'Ищет работу на должность:'

        def is_it_role(value: str) -> bool:
            """Check if a job title string corresponds to an IT role."""
            if not isinstance(value, str):
                return False
            val = value.lower()

            if any(kw in val for kw in self._STRONG_KEYWORDS):
                return True
            if any(kw in val for kw in self._TECH_KEYWORDS):
                return True

            if any(kw in val for kw in self._WEAK_KEYWORDS):
                return any(ctx_kw in val for ctx_kw in self._IT_CONTEXT)

            return False

        df = df[df[col].apply(is_it_role)]
        
        logging.info(f"FilterITRolesHandler: {initial_count} -> {len(df)} rows ({len(df)/initial_count*100:.1f}%)")
        ctx.df = df
        return ctx
