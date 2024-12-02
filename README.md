# Personal Blog/Website


This repository was created to embrace the "learn by doing" approach, which is my most efficient learning style. I am creating my personal blog/website using the [Rio-UI codebase](https://github.com/rio-labs/rio), which is a library for building full-stack web apps entirely in pure Python.


### Key Features of rio-ui

- **Modern, Declarative UI Framework:**

\u2217 **100% Python:** No need for HTML, CSS, or JavaScript.
\u2217 **50+ Built-in Components:** Easily create common UI elements like switches, buttons, and text fields.
\u2217 **Seamless Integration with Modern Python Tooling:** Enjoy instant suggestions and error highlighting with type safety.
\u2217 **Built-in Developer Tools:** Simplify your development workflow.

- **Open Source & Free foreve**

### How to Run the Code

```bash
git clone https://github.com/linhduongtuan/personal_blog.git
cd personal_blog
```
```python 
# It's recommended to use 'uv' for creating virtual environments and installing packages swiftly
pip install uv
uv venv --python 3.13
source .venv/bin/activate # for MacOS and Linux
# .venv\Scripts\activate for Windows

uv pip install -r requirements.txt

# then run
cd personal-blog
rio run
# if error, please run
# python -m rio run
```


### TODO:
- [ ] Make login, logout and signout functions

- [ ] Implement a backend to store data in a database

- [ ] Polish both the code and the web interface

- [ ] Create nested pages

- [ ] Implement a working search function

- [ ] Add more realistic features to make the website fully functional

- [ ] And so on

### Random Thoughts:

- I am not sure how difficult it is to develop a full-stack website based on other codebases or frameworks. Can you tell me more about your experience in web development?

- Indeed, I love library [rio-ui](https://github.com/rio-labs/rio) because it compiles and renders web interfaces much faster than that of `Reflex`. However, **rio-ui** is still not as mature as **Reflex** and lacks certain tutorials, which makes it harder for me to learn.

- Similarly, I also create my own personal blog using **Reflex**. Please refer [https://github.com/linhduongtuan/portfolio_web](https://github.com/linhduongtuan/portfolio_web) and **be generous to give me a star**

### References:

- [Official Rio-UI Documentation](https://rio.dev/docs)

- [Astral uv](https://docs.astral.sh/uv/): [An extremely fast Python package and project manager](https://github.com/astral-sh/uv), written in Rust.

- Special thanks to the [Abacus.ai platform](https://apps.abacus.ai/) for helping with creating and debugging code.
