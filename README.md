# EleutherAI Website 2021

This is the new website for EleutherAI based on Hugo, a static site generator. The content should correspond to existing google sites website but with added blog and other features. Please make sure to familiarize with the basics of working with Hugo before you start using it.


## How it works?
1. [install hugo](https://gohugo.io/getting-started/installing/)
2. clone this repository (and make sure you are in the `master` branch)
3. get git submodules which serve as the generated website in the public folders (/public, /public-blog ...etc): `git submodule update --init` and then make sure that you have set them all to `master` branch.
4. now you can try to run hugo locally `hugo server -D`

if everything is working feel at this point feel free to start working on the website. 

Once you are done with your work you cant now try to  publish the changes. To do that you need to push the git changes both for the repo and for /public folder that is set as git submodule for the website html content.

Easiest way to do it is to run deploy scripts:
`./deploy.sh` and then just commit changes frgom the main repo.

If that doesnt work you can do it manually:

1. go to public folder `cd public`
2. commit the changes and push them. `git add`, `git commit -m [commit name for submodule]`, `git push`
3. go back to the main repo `cd ..` and commit+push your changes there too `git add`, `git commit -m [commit name for project]`, `git push`

***Note: based on your user settings you might not have privileges to do changes in /public folder. In that case you can still do all the previous steps with `sudo` command.***

So it will be `sudo ./deploy.sh`, `sudo git commit `.. etc.

## Blog
To use the blog, the instructions are similar like in the previous section with few differences. The blog markdown content is served from content-blog, the submodule repo is public-blog and it uses different config file: config-blog.toml and different deploy script deploy_blog.sh.

1. change the content in content-blog
2. when deploying run: `./deploy_blog.sh`
3. commit/push the changes for the main rep

### Update: now there is the script that should do all of this for you. You just need to run:

`./run_all.sh` 

***(before running the script make sure all git submodules are in master branch as otherwise it wont push)***

### and it will generate all the sites and deploy them. (both for website and blog).

## Editing content
The theme and content structure should be similar to the standard Hugo projects. Content for main pages is in `/content folder`. The Blog is in `/content/blog`. 

`content/projects` is the markdown content for the project pages (GPT-NEO, The pile), `content/project-intros` are small chunks of the project contents that are displayed on the home page.

### How to display 2 containers that are horizontally aligned?

1. add an empty header markdown with the class `content-block` -> `##  ## {class="content-block"}`
2. after that line, add 2 containers as 2 elements of the list. In CSS it is defined that first list `<ul>` below content-block header will display items horizontally. (only the first one, any other list elements will be displayed as expected)
