"""Data filtering transformations."""

import logging

from src.core import DataContainer, DataStage

logger = logging.getLogger(__name__)


class ITDeveloperFilterStage(DataStage):
    """Stage for filtering IT developer resumes."""
    
    # Primary IT role keywords
    PRIMARY_KEYWORDS = [
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
    
    # Technology stack indicators
    TECH_STACK_KEYWORDS = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
        'golang', 'go ', 'rust', 'ruby', 'php', 'swift', 'kotlin',
        'react', 'angular', 'vue', 'django', 'flask', 'spring',
        'docker', 'kubernetes', 'k8s', 'terraform', 'ansible',
        'aws', 'azure', 'gcp', 'sql', 'nosql', 'mongodb', 'postgresql',
        'linux', 'unix',
        'api', 'microservice', 'микросервис',
    ]
    
    # Ambiguous keywords requiring IT context
    AMBIGUOUS_KEYWORDS = [
        'инженер', 'engineer', 'администратор', 'admin',
        'аналитик', 'analyst', 'архитектор', 'architect',
        'консультант', 'consultant', 'менеджер проект', 'project manager',
    ]
    
    # IT context indicators
    IT_CONTEXT_INDICATORS = [
        'по', 'программ', 'софт', 'soft', 'it', 'ит',
        'информац', 'автоматиз', 'асу', 'erp', 'crm', 'sap',
        'devops', 'cloud', 'облач', 'сет', 'network', 'систем',
        'данн', 'data', 'баз', 'database', 'cyber', 'кибер',
        'техн', 'tech', 'цифр', 'digital',
    ]
    
    def __init__(self):
        """Initialize IT developer filter."""
        super().__init__("ITDeveloperFilter")
    
    def _is_it_position(self, job_title: str) -> bool:
        """
        Determine if a job title represents an IT position.
        
        Args:
            job_title: Job title string to check.
            
        Returns:
            True if position is IT-related, False otherwise.
        """
        if not isinstance(job_title, str):
            return False
        
        title_lower = job_title.lower()
        
        # Check primary keywords
        if any(keyword in title_lower for keyword in self.PRIMARY_KEYWORDS):
            return True
        
        # Check tech stack keywords
        if any(keyword in title_lower for keyword in self.TECH_STACK_KEYWORDS):
            return True
        
        # Check ambiguous keywords with context
        if any(keyword in title_lower for keyword in self.AMBIGUOUS_KEYWORDS):
            return any(context in title_lower for context in self.IT_CONTEXT_INDICATORS)
        
        return False
    
    def execute(self, container: DataContainer) -> DataContainer:
        """Filter dataset to IT developers only."""
        df = container.processed_data
        initial_size = len(df)
        
        job_title_column = 'Ищет работу на должность:'
        mask = df[job_title_column].apply(self._is_it_position)
        container.processed_data = df[mask].copy()
        
        filtered_size = len(container.processed_data)
        retention_rate = (filtered_size / initial_size * 100) if initial_size > 0 else 0
        
        logger.info(
            f"Filtered dataset: {initial_size} -> {filtered_size} rows "
            f"({retention_rate:.1f}% retained)"
        )
        
        return container

