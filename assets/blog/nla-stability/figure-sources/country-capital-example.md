# Example explanation edits: country-capital item 0

Base query 'China' (answer 'Beijing') -> target query 'Chile' (answer 'Santiago'). Raw-text prompt, layer-32 NLA; the explanation is the AV's account of the final prompt token (`:`).

## Seed 0

Unpatched answer: 'Beijing'

| arm | answer, final token | answer, last four tokens |
|---|---|---|
| Query-and-answer substitution | 'Santiago' | 'Santiago' |
| Final-paragraph rewrite | 'Beijing' | 'Lima' |
| Combined edit | 'Santiago' | 'Santiago' |
| Donor patching | 'Santiago' | 'Santiago' |

### Original explanation

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Beijing" is a standard geographic label, requiring the name of China's capital to complete the answer — "Beijing" is the city name for China.

Final token "are:" ends a location list ("the Republic and..."), requiring a city name or capitalized country label — almost certainly "Beijing" or "Beijing, China" — completing the "Capital:..." answer format, likely followed by "Beijing" or "Beijing, the capital." or "Shanghai" to name the capital city of China's government.
```

### Query-and-answer substitution

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Santiago" is a standard geographic label, requiring the name of Chile's capital to complete the answer — "Santiago" is the city name for Chile.

Final token "are:" ends a location list ("the Republic and..."), requiring a city name or capitalized country label — almost certainly "Santiago" or "Santiago, Chile" — completing the "Capital:..." answer format, likely followed by "Santiago" or "Santiago, the capital." or "Shanghai" to name the capital city of Chile's government.
```

### Final-paragraph rewrite

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Beijing" is a standard geographic label, requiring the name of China's capital to complete the answer — "Beijing" is the city name for China.

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

### Combined edit

```
Structured FAQ format with factual geographic/demographic content establishes a list of country flags and capitals, following consistent pattern.

The phrase "Capital: Santiago" is a standard geographic label, requiring the name of Chile's capital to complete the answer — "Santiago" is the city name for Chile.

Final token "are:" begins the answer field of this paired-response format, requiring the completion "Santiago" to give the expected next token. or "Santiago" or "Santiago" or "Santiago".
```

