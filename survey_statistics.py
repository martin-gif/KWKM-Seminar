
import pandas as pd

# Use the same data file and column names as correlation.py
DATA_FILE = "../data/results-survey374736.csv"  # eure Antworten (25 x 74)


class SurveyStatistics:
    def __init__(self, csv_path=DATA_FILE):
        self.data = pd.read_csv(csv_path)

    def age_statistics(self):
        age_col = "G02Q04"
        if age_col in self.data.columns:
            age_series = pd.to_numeric(self.data[age_col], errors="coerce")
            age_counts = age_series.value_counts().sort_index().to_dict()
            return {
                'mean': age_series.mean(),
                'median': age_series.median(),
                'std': age_series.std(),
                'min': age_series.min(),
                'max': age_series.max(),
                'count': age_series.count(),
                'participants_per_age': age_counts
            }
        else:
            return f'Column "{age_col}" not found.'

    def gender_statistics(self):
        # Gender: G02Q05 and G02Q05[other]
        gender_cols = ["G02Q05", "G02Q05[other]"]
        found = [col for col in gender_cols if col in self.data.columns]
        if not found:
            return 'No gender columns found (expected G02Q05, G02Q05[other]).'
        # Combine both columns for total counts
        gender_data = self.data[found].fillna('')
        # If both columns, merge into one Series
        combined = gender_data.apply(lambda row: row[0] if row[0] else row[1], axis=1)
        gender_counts = combined.value_counts().to_dict()
        return {
            'participants_per_gender': gender_counts,
            'total': combined.count(),
            'columns_used': found
        }

    def school_education_statistics(self):
        # G02Q06 and G02Q06[other]
        school_cols = ["G02Q06", "G02Q06[other]"]
        found = [col for col in school_cols if col in self.data.columns]
        if not found:
            return 'No school education columns found (expected G02Q06, G02Q06[other]).'
        school_data = self.data[found].fillna('')
        combined = school_data.apply(lambda row: row[0] if row[0] else row[1], axis=1)
        school_counts = combined.value_counts().to_dict()
        return {
            'participants_per_school_education': school_counts,
            'total': combined.count(),
            'columns_used': found
        }

    def vocational_education_statistics(self):
        # G02Q07 and G02Q07[other]
        voc_cols = ["G02Q07", "G02Q07[other]"]
        found = [col for col in voc_cols if col in self.data.columns]
        if not found:
            return 'No vocational/academic education columns found (expected G02Q07, G02Q07[other]).'
        voc_data = self.data[found].fillna('')
        combined = voc_data.apply(lambda row: row[0] if row[0] else row[1], axis=1)
        voc_counts = combined.value_counts().to_dict()
        return {
            'participants_per_vocational_education': voc_counts,
            'total': combined.count(),
            'columns_used': found
        }

    def summary(self):
        return {
            'age': self.age_statistics(),
            'gender': self.gender_statistics(),
            'school_education': self.school_education_statistics(),
            'vocational_education': self.vocational_education_statistics()
        }

if __name__ == "__main__":
    stats = SurveyStatistics()
    print("Age Statistics:", stats.age_statistics())
    print("Gender Statistics:", stats.gender_statistics())
    print("Bildungsstand Statistics:", stats.bildungsstand_statistics())
    print("Summary:", stats.summary())
