# EleutherAI Website 2021

This is the new website for EleutherAI based on Hugo, a static site generator. The content should correspond to existing google sites website but with added blog and other features. Please make sure to familiarize with the basics of working with Hugo before you start using it.

## Setup
1. [install hugo](https://gohugo.io/getting-started/installing/)
2. clone this repository (and make sure you are in the `master` branch)
3. get git submodules which serve as the generated website in the public folders (/public, /public-blog ...etc): `git submodule update --init` and then make sure that you have set them all to `master` branch.

## Project Layout

| Directory      | Description |
| -----------: | ----------- |
| `content` | Underyling content for the main site|
| `content-blog` | Underyling content for the blog |
| `static/images` | Images for both sites. |
| `themes/eai-theme` | We use a single theme for both the main site and the blog. | 
| `public` | Contains the main site build | 
| `public-blog` | Build for the blog.| 

## How to display 2 containers that are horizontally aligned?

1. add an empty header markdown with the class `content-block` -> `##  ## {class="content-block"}`
2. after that line, add 2 containers as 2 elements of the list. In CSS it is defined that first list `<ul>` below content-block header will display items horizontally. (only the first one, any other list elements will be displayed as expected)

## Dev Environment

To run the development server on localhost for the main site:

`hugo server -D`

To load the blog instead:

`hugo server -D --config config-blog.toml`

To bind on another IP apart from localhost and change the baseURL (ensuring the links work):

`hugo server --bind=BIND-IP --baseUrl=IP-OR-DOMAIN -D`

If everything is working at this point feel free to start working on the website. Once you are happy with the changes, perform the build as explained below.

## Building And Pushing

We are using submodules for the site builds (public and public-blog) so these need to be built and pushed separately to the underlying template and content changes.

We have created build scripts to make this process easier:

**Main Site:** `./deploy.sh` 
**Blog:** `./deploy_blog.sh` 
**Both:** `./run_all.sh` 

Afterwards you can separately push your underlying changes.

***Note: based on your user settings you might not have privileges to do changes in /public folder. In that case you can still do all the previous steps with `sudo` command.***

So it will be `sudo ./deploy.sh`, `sudo git commit `.. etc.

***(before running the script make sure all git submodules are in master branch as otherwise it wont push)***




