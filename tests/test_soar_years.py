import unittest

from generate_hugo_data import normalize_paper_row, soar_paper_years


class SoarYearsTests(unittest.TestCase):
    def test_program_year_not_publication_year(self):
        rows = [normalize_paper_row({
            "Title": "Earlier cohort, later publication",
            "Date": "2026-01-01",
            "Additional Metadata": "SOAR 2025",
        }), normalize_paper_row({
            "Title": "New cohort",
            "Additional Metadata": "SOAR 2026",
        }), normalize_paper_row({
            "Title": "Not SOAR",
            "Date": "2025-01-01",
        })]
        groups = soar_paper_years(rows)
        self.assertEqual([group["year"] for group in groups], ["2026", "2025"])
        self.assertEqual(groups[1]["papers"][0]["title"], "Earlier cohort, later publication")
        self.assertEqual(len(groups[0]["papers"]), 1)


if __name__ == "__main__":
    unittest.main()
