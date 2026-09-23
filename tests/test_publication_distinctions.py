import unittest

from generate_hugo_data import area_paper_record, library_paper_record


class PublicationDistinctionTests(unittest.TestCase):
    def test_superlative_is_independent_of_venue_and_status(self):
        award = "Oral at MATH-AI Workshop, NeurIPS"
        cases = [
            ("Accepted", "NeurIPS", "", "NeurIPS"),
            ("Accepted", "NeurIPS", "MATH-AI Workshop, NeurIPS", "NeurIPS"),
            ("Accepted", "", "MATH-AI Workshop, NeurIPS", "MATH-AI Workshop, NeurIPS"),
            ("Preprint", "", "", "arXiv"),
            ("Under Review", "NeurIPS", "", "arXiv"),
        ]
        for status, conference, workshop, expected_venue in cases:
            with self.subTest(status=status, conference=conference, workshop=workshop):
                row = {
                    "Title": "Test Paper",
                    "Pub Date": "Sep 21, 2026",
                    "Status": status,
                    "Conference or Journal": conference,
                    "Workshop": workshop,
                    "Superlative": award,
                }
                paper = library_paper_record(row)
                self.assertEqual(paper["venue"], expected_venue)
                self.assertEqual(paper["superlatives"], [award])
                self.assertEqual(paper["marker"], "oral")
                self.assertEqual(paper["marker"], area_paper_record(row)["marker"])

    def test_award_wording_and_existing_marker_priority_are_preserved(self):
        cases = [
            ("  Workshop Best Paper Runner-up  ", "runnerup"),
            ("Oral; Best Paper", "bestpaper"),
            ("SPOTLIGHT at a workshop", "spotlight"),
            ("Outstanding Paper", "bestpaper"),
            ("Finalist and Oral", "runnerup"),
            ("Honorable Mention", ""),
            ("", ""),
        ]
        for award, marker in cases:
            with self.subTest(award=award):
                paper = library_paper_record({"Superlative": award})
                self.assertEqual(paper["superlatives"], [award.strip()] if award.strip() else [])
                self.assertEqual(paper["marker"], marker)

    def test_venue_does_not_supply_an_unlisted_award(self):
        paper = library_paper_record({
            "Status": "Accepted",
            "Workshop": "MATH-AI Workshop, Best Paper",
            "Superlative": "",
        })
        self.assertEqual(paper["superlatives"], [])
        self.assertEqual(paper["marker"], "")

    def test_first_place_uses_its_own_marker_and_preserves_wording(self):
        for award in ["First Place at the Symbolic Music Generation competition", "FIRST PLACE"]:
            with self.subTest(award=award):
                row = {"Superlative": award}
                paper = library_paper_record(row)
                self.assertEqual(paper["superlatives"], [award])
                self.assertEqual(paper["marker"], "firstplace")
                self.assertEqual(area_paper_record(row)["marker"], "firstplace")

    def test_legacy_superlatives_column_still_works(self):
        paper = library_paper_record({"Superlatives": "Oral; Best Paper"})
        self.assertEqual(paper["superlatives"], ["Oral", "Best Paper"])
        self.assertEqual(paper["marker"], "bestpaper")


if __name__ == "__main__":
    unittest.main()
