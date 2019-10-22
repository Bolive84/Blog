# Instructions

*Windows*

>  You will need to use a combination of gitbash (GB) + Anaconda Prompt (AP)

- `dir` = `ls` 
- `cd` = `cd`

*Mac and Linux*

>  Just use Terminal

---

#### Initial Setup

1. Create a new github Repo (call it **blog**), and initialize with python .gitignore

2. Download it to your computer [GB]

   ```
   git clone <repo>
   cd <repo>
   ```

3. Create a bunch of empty folders [GB]

   ```
   mkdir content jupyter output theme
   mkdir content/images jupyter/images theme/templates
   ```

4. Create a requirements file [GB]

   ```
   touch requirements.txt
   ```

5. Fill `requirements.txt` with [Atom]

   ```
   pelican
   Markdown
   jupyter
   ghp-import
   ```

6. Install everything [AP]

   ```
   pip install -r requirements.txt
   ```

7. Create a pelican configuration file [GB]

   ```
   touch pelicanconf.py
   ```

8. Fill the `pelicanconf.py` with [Atom]

   ```
   #!/usr/bin/env python
   # -*- coding: utf-8 -*- #
   from __future__ import unicode_literals
   import os
   
   AUTHOR = 'Max Humber'
   SITENAME = 'yeti'
   SITEURL = 'https://maxhumber.github.io/yeti'
   # SITEURL = 'https://pages.git.generalassemb.ly/max/yeti'
   PATH = 'content'
   STATIC_PATHS = ['images']
   TIMEZONE = 'America/Toronto'
   DEFAULT_LANG = 'en'
   DEFAULT_PAGINATION = 10
   RELATIVE_URLS = True
   ```

#### Create a Blog Post

9. Create a dummy blog post [GB]

   ```
   touch content/blog_1.md
   ```

10. Fill `blog_1.md` with [Atom]

    ```
    Title: My First Blog Post!
    Date: 2019-09-12 10:30
    Slug: blog1
    
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt ornare massa. Cursus vitae congue mauris rhoncus. Pulvinar neque laoreet suspendisse interdum. Eu nisl nunc mi ipsum faucibus vitae. Sapien faucibus et molestie ac feugiat sed lectus vestibulum mattis. Nibh tortor id aliquet lectus proin nibh nisl. Sit amet venenatis urna cursus eget. Amet consectetur adipiscing elit duis. Quam pellentesque nec nam aliquam sem et tortor. Congue nisi vitae suscipit tellus.
    ```

11. Generate the HTML artifacts [AP]

    ```
    pelican -s pelicanconf.py -o output content
    ```

12. Test it locally [AP]

    ```
    cd output; python -m http.server
    ```

#### First Deploy

- Deploy for Mac

  ````
  ghp-import -m "Generate Pelican site" -b gh-pages output
  git push origin gh-pages
  ````

- Deploy for Windows [GB]

  ```
  git add .
  git commit -m 'new blog post'
  git push
  git subtree push --prefix output origin gh-pages
  ```

#### Less Ugly 

13. Go into the templates folder and create three empty files [GB]

    ```
    cd theme/templates
    touch base.html index.html article.html
    ```

14. Fill `base.html` [Atom]

    ```
    <!DOCTYPE html>
    <html lang="en">
    <head>
     <title>{% block title %}{% endblock %}</title>
     <!-- Latest compiled and minified CSS -->
     <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css" integrity="sha384-HSMxcRTRxnN+Bdg0JdbxYKrThecOKuH5zCYotlSAcp1+c8xmyTe9GYg1l9a69psu" crossorigin="anonymous">
     <div class="container">
      <h1><a href="{{ SITEURL }}">{{ SITENAME }}</a></h1>
     </div>
    </head>
    <body>
     <div class="container">
      {% block content %}{% endblock %}
     </div>
    </body>
    </html>
    ```

15. Fill `index.html` [Atom]

    ```
    {% extends "base.html" %}
    {% block title %}{{ SITENAME }}{% endblock %}
    {% block content %}
    <div class="row">
     <div class="col-md-8">
      {% for article in articles %}
       <h3><a href="{{ SITEURL }}/{{ article.slug }}.html">{{ article.title }}</a></h3>
       <label>{{ article.date.strftime('%Y-%m-%d') }}</label>
      {% else %}
       No posts yet!
      {% endfor %}
     </div>
    </div>
    {% endblock %}
    ```

16. Fill `article.html`

    ```
    {% extends "base.html" %}
    {% block title %}{{ article.title }}{% endblock %}
    {% block content %}
    <div class="row">
     <div class="col-md-8">
      <h3>{{ article.title }}</h3>
      <label>{{ article.date.strftime('%Y-%m-%d') }}</label>
      {{ article.content }}
     </div>
    </div>
    {% endblock %}
    ```

17. Now we need to clean up the ugly default theme [GB]

    ```
    rm -rf output; mkdir output
    ```

18. And regenerate the HTML artifacts [AP]

    ```
    pelican -s pelicanconf.py -o output -t theme content
    ```

- Deploy for Mac

  ```
  ghp-import -m "Generate Pelican site" -b gh-pages output
  git push origin gh-pages
  ```

- Deploy for Windows [GB]

  ```
  git add .
  git commit -m 'new blog post'
  git push
  git subtree push --prefix output origin gh-pages
  ```

#### Jupyter to Blog

- Create a Jupyter notebook

- Export to markdown (.md)

- Move the markdown file to the `content` folder

- Add these three lines to the very top of the generated markdown

  ```
  Title: My Second Blog
  Date: 2019-09-12 11:35
  Slug: blog2
  ```

- Generate the website with Pelican [AP]

  ```
  pelican -s pelicanconf.py -o output -t theme content
  ```

- Deploy Mac

  ```
  ghp-import -m "Generate Pelican site" -b gh-pages output
  git push origin gh-pages
  ```

- Deploy Windows [GB]

  ```
  git add .
  git commit -m 'new blog post'
  git push
  git subtree push --prefix output origin gh-pages
  ```