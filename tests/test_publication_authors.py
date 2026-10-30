import unittest
from unittest.mock import mock_open, patch

from generate_hugo_data import full_author_terms, library_paper_record, read_papers


class PublicationAuthorDisplayTests(unittest.TestCase):
    def test_missing_optional_authors_do_not_block(self):
        csv = 'Title,Date,Display Authors,all authors,EAI Affils\nExample,"October 30, 2026",Smith and Jones,,\n'
        with patch('pathlib.Path.open', mock_open(read_data=csv)):
            self.assertEqual(len(read_papers()), 1)

    def test_missing_essential_fields_are_skipped(self):
        csv = 'Title,Date,Display Authors\n,"October 30, 2026",Smith\nExample,,Smith\nExample,"October 30, 2026",\n'
        with patch('pathlib.Path.open', mock_open(read_data=csv)):
            self.assertEqual(read_papers(), [])

    def test_comma_separated_full_names(self):
        names = "Vilém Zouhar, Hanna Yukhymenko, Stella Biderman"
        self.assertEqual(full_author_terms(names), ["Vilém Zouhar", "Hanna Yukhymenko", "Stella Biderman"])
        paper = self.paper("Zouhar and others", names)
        self.assertEqual(paper["authors"], "Zouhar and others")
        for name in full_author_terms(names):
            self.assertIn(name, paper["author_search"])

    def test_legacy_surname_first_semicolon_list(self):
        self.assertEqual(full_author_terms("Zouhar, Vilém; Biderman, Stella"),
                         ["Zouhar, Vilém", "Biderman, Stella"])

    def test_empty_full_names(self):
        self.assertEqual(full_author_terms(None), [])
        self.assertEqual(full_author_terms(" , , "), [])

    def paper(self, display_authors, all_authors):
        return library_paper_record(
            {
                "Title": "Test Paper",
                "Date": "Aug 13, 2026",
                "Display Authors": display_authors,
                "all authors": all_authors,
            }
        )

    def test_display_authors_are_preserved(self):
        cases = [
            ("Chang, Arnett, and 378 others", "Tyler A. Chang; Catherine Arnett"),
            ("Zhang and Goldstein", "Ruichong Zhang; Daniel Goldstein"),
            (
                "Son, Kim, Arnett, Ko, Lee, Kang, Jiang, Yun, Lee, and 67 others",
                "Guijin Son; Seungone Kim; Catherine Arnett; Hyunwoo Ko",
            ),
            (
                "O'Mahony, Grinsztajn, Schoelkopf, and Biderman",
                "Laura O'Mahony; Stella Biderman",
            ),
            ("Laurençon, Névéol, Ilić, and Suárez", "Hugo Laurençon; Aurélie Névéol"),
            ("Komatsuzaki", "Aran Komatsuzaki"),
        ]

        for display_authors, all_authors in cases:
            with self.subTest(display_authors=display_authors):
                paper = self.paper(display_authors, all_authors)
                self.assertEqual(paper["authors"], display_authors)
                self.assertTrue(
                    all(author in paper["author_search"] for author in all_authors.split("; "))
                )

    def test_only_outer_whitespace_is_removed(self):
        paper = self.paper(
            "  Paulo, Marshall and Belrose  ",
            "Gonçalo Paulo; Jack Marshall; Nora Belrose",
        )

        self.assertEqual(paper["authors"], "Paulo, Marshall and Belrose")


if __name__ == "__main__":
    unittest.main()
