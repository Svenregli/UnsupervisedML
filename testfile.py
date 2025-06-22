# Quick test
from ul_pipeline import AcademicResearchAnalyzer

# Test with your existing data
analyzer = AcademicResearchAnalyzer()
result = analyzer.perform_full_analysis_with_auto_save("C:\\Users\\sven-\\OneDrive\\Dokumente\\UNI LU\\Master\\Master\\FS25\\UnsupervisedML\\data\\openalex_cache\\search_artificial_intelligence_200.json", "search_artificial_intelligence_200")

print(f"Analysis completed: {result}")