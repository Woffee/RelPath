# RelPath

## Prerequisites

```sh
conda create --prefix py10 python=3.10
conda activate py10
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib==3.8.3
pip install scipy==1.12.0
pip install gitpython==3.1.41
pip install tensorboard==2.16.1
```

## How to run the code

```sh
python train.py
```

## Spacy Entities

- PERSON - People, including fictional.
- NORP - Nationalities or religious or political groups.
- FAC - Buildings, airports, highways, bridges, etc.
- ORG - Companies, agencies, institutions, etc.
- GPE - Countries, cities, states.
- LOC - Non-GPE locations, mountain ranges, bodies of water.
- PRODUCT - Objects, vehicles, foods, etc. (Not services.)
- EVENT - Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART - Titles of books, songs, etc.
- LAW - Named documents made into laws.
- LANGUAGE - Any named language.
- DATE - Absolute or relative dates or periods.
- TIME - Times smaller than a day.
- PERCENT - Percentage, including "%".
- MONEY - Monetary values, including unit.
- QUANTITY - Measurements, as of weight or distance.
- ORDINAL - "first", "second", etc.
- CARDINAL - Numerals that do not fall under another type.


## License

Distributed under the MIT License.




