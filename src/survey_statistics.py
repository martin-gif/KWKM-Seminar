import pandas as pd
import numpy as np


class SurveyStatistics:
    """
    Analyzes survey data from demographic questions.
    Accepts DataFrames directly instead of file paths.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: pandas DataFrame containing survey data
        """
        self.data = df

    def age_statistics(self):
        age_col = "G02Q04"
        if age_col in self.data.columns:
            age_series = pd.to_numeric(self.data[age_col], errors="coerce")
            age_counts = age_series.value_counts().sort_index().to_dict()
            return {
                'mean': float(age_series.mean()),
                'median': float(age_series.median()),
                'std': float(age_series.std()),
                'min': int(age_series.min()),
                'max': int(age_series.max()),
                'count': int(age_series.count()),
                'participants_per_age': {int(k): int(v) for k, v in age_counts.items()}
            }
        else:
            return f'Column "{age_col}" not found.'

    def gender_statistics(self):
        # Gender: G02Q05 and G02Q05[other]
        gender_cols = ["G02Q05", "G02Q05[other]"]
        found = [col for col in gender_cols if col in self.data.columns]
        if not found:
            return 'No gender columns found.'
        # Combine both columns for total counts
        gender_data = self.data[found].fillna('')
        # Use .iloc to avoid FutureWarning
        combined = gender_data.apply(lambda row: row.iloc[0] if row.iloc[0] else row.iloc[1], axis=1)
        gender_counts = combined.value_counts().to_dict()
        
        gender_mapping = {
            '1': 'Male', '2': 'Female', '3': 'Other',
            1: 'Male', 2: 'Female', 3: 'Other',
        }
        mapped_counts = {gender_mapping.get(k, f'Unknown ({k})'): int(v) for k, v in gender_counts.items()}
        
        return {
            'participants_per_gender': mapped_counts,
            'total': int(combined.count()),
        }

    def school_education_statistics(self):
        # G02Q06 and G02Q06[other]
        school_cols = ["G02Q06", "G02Q06[other]"]
        found = [col for col in school_cols if col in self.data.columns]
        if not found:
            return 'No school education columns found.'
        school_data = self.data[found].fillna('')
        combined = school_data.apply(lambda row: row.iloc[0] if row.iloc[0] else row.iloc[1], axis=1)
        school_counts = combined.value_counts().to_dict()
        
        school_mapping = {
            '1': 'Primary', '2': 'Secondary', '3': 'Tertiary', '4': 'University',
            1: 'Primary', 2: 'Secondary', 3: 'Tertiary', 4: 'University',
        }
        mapped_counts = {school_mapping.get(k, f'Unknown ({k})'): int(v) for k, v in school_counts.items()}
        
        return {
            'participants_per_school_education': mapped_counts,
            'total': int(combined.count()),
        }

    def vocational_education_statistics(self):
        # G02Q07 and G02Q07[other]
        voc_cols = ["G02Q07", "G02Q07[other]"]
        found = [col for col in voc_cols if col in self.data.columns]
        if not found:
            return 'No vocational education columns found.'
        voc_data = self.data[found].fillna('')
        combined = voc_data.apply(lambda row: row.iloc[0] if row.iloc[0] else row.iloc[1], axis=1)
        voc_counts = combined.value_counts().to_dict()
        
        voc_mapping = {
            '1': 'No Training', '2': 'In Training', '3': 'Completed', '4': 'Advanced', '5': 'Specialized',
            1: 'No Training', 2: 'In Training', 3: 'Completed', 4: 'Advanced', 5: 'Specialized',
        }
        mapped_counts = {voc_mapping.get(k, f'Unknown ({k})'): int(v) for k, v in voc_counts.items()}
        
        return {
            'participants_per_vocational_education': mapped_counts,
            'total': int(combined.count()),
        }

    def summary(self):
        return {
            'age': self.age_statistics(),
            'gender': self.gender_statistics(),
            'school_education': self.school_education_statistics(),
            'vocational_education': self.vocational_education_statistics()
        }

    def print_summary(self):
        """Print all survey statistics in formatted tables."""
        print("\n" + "=" * 80)
        print(" " * 20 + "SURVEY STATISTICS SUMMARY")
        print("=" * 80)

        # Age Statistics
        age_stats = self.age_statistics()
        print("\n📊 AGE STATISTICS")
        print("-" * 80)
        age_table = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Total Participants"],
            "Value": [
                f"{age_stats['mean']:.2f}",
                f"{age_stats['median']:.1f}",
                f"{age_stats['std']:.2f}",
                age_stats['min'],
                age_stats['max'],
                age_stats['count'],
            ],
        })
        print(age_table.to_string(index=False))

        print("\n📈 Age Distribution:")
        age_dist_df = pd.DataFrame(
            list(age_stats["participants_per_age"].items()),
            columns=["Age", "Count"],
        ).sort_values("Age").reset_index(drop=True)
        print(age_dist_df.to_string(index=False))

        # Gender Statistics
        print("\n" + "-" * 80)
        print("👥 GENDER STATISTICS")
        print("-" * 80)
        gender_stats = self.gender_statistics()
        gender_df = pd.DataFrame(
            list(gender_stats["participants_per_gender"].items()),
            columns=["Gender", "Count"],
        ).reset_index(drop=True)
        print(gender_df.to_string(index=False))
        print(f"\nTotal Respondents: {gender_stats['total']}")

        # School Education Statistics
        print("\n" + "-" * 80)
        print("🎓 SCHOOL EDUCATION LEVEL")
        print("-" * 80)
        school_stats = self.school_education_statistics()
        school_df = pd.DataFrame(
            list(school_stats["participants_per_school_education"].items()),
            columns=["Education Level", "Count"],
        ).reset_index(drop=True)
        print(school_df.to_string(index=False))
        print(f"\nTotal Respondents: {school_stats['total']}")

        # Vocational Education Statistics
        print("\n" + "-" * 80)
        print("🛠️  VOCATIONAL EDUCATION")
        print("-" * 80)
        voc_stats = self.vocational_education_statistics()
        voc_df = pd.DataFrame(
            list(voc_stats["participants_per_vocational_education"].items()),
            columns=["Vocational Status", "Count"],
        ).reset_index(drop=True)
        print(voc_df.to_string(index=False))
        print(f"\nTotal Respondents: {voc_stats['total']}")

        print("\n" + "=" * 80 + "\n")
