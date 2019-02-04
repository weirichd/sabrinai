from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

import os


def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


sabrina_url = 'https://sabrinatranscripts.wordpress.com/'

sabrina_html = simple_get(sabrina_url)

bs = BeautifulSoup(sabrina_html, 'html.parser')

entry = bs.find(class_='entry')

# Not looking for the links to the seasons.
seasons = [ep for ep in entry.findChildren('p') if not ep.text.strip().startswith('Season')]

episodes = []

for season in seasons:
    episodes += [(a['href'], a.text) for a in season.findChildren('a')]

for i, (url, name) in enumerate(episodes):
    episode_html = simple_get(url)

    filename = '{:03}_{}'.format(i, ''.join(c for c in name if c.isalpha()).lower())

    bs = BeautifulSoup(episode_html, 'html.parser')

    print('{}/{}: {}'.format(i, len(episodes), name))

    page_text = [p.text.strip() for p in bs.find_all('p')]

    header_end_index = page_text.index(next(x for x in page_text if x.startswith('DISCLAIMER')))

    footer_begin_index = page_text.index(next(x for x in page_text[::-1] if x.startswith('This entry was posted on')))

    if not os.path.exists('episodes'):
        os.makedirs('episodes')

    with open('episodes/' + filename + '.txt', 'w') as f:
        for p in page_text[header_end_index + 1:footer_begin_index]:
            f.write(p + '\n')

print('Done! :)')
