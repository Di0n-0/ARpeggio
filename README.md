### ARpeggio
[![ARpeggio Showcase Video](http://img.youtube.com/vi/je6cG32vBNg/0.jpg)](http://www.youtube.com/watch?v=je6cG32vBNg)

Do you find reading sheet music or tablatures a bit daunting? Well, you're not alone. I'll be honest, I'm not the most consistent player, and I often go through extended periods without practice. It can be frustrating to pick up where I left off with reading sheet music or tablatures every time I start playing again. That's why I developed this program: to tackle this annoyance head-on.

Introducing ARpeggio, a Python-based program that I'm considering rewriting in C/C++ before publishing. With the help of AI, it's designed to teach you how to play the guitar. I've been working on it for almost two months now, and I'm excited to share it with you.

ARpeggio is all about making guitar learning more accessible and enjoyable. No need to worry about deciphering complicated notations. This program aims to guide you through your guitar journey without relying on traditional sheet music or tablatures. It's like having your very own personal guitar instructor right at your fingertips.

I'm proud of the progress I've made with ARpeggio, and I can't wait to bring it to you. Stay tuned for updates on its development and release on my website.

**Features**
- Option to only display chosen hand indices
- Capability to adapt to the rotation of your guitar and to the distance between it and the camera
- Capability to process *any* video and extract its hand landmarks for you to replay on your guitar, as long as *The 1 commandment and 2 suggestions* are abode
  - *Thou shalt film videos from one angle only! No wandering cameras shall be permitted! Let the visuals remain constant, granting thou a sense of stability and coherence!*
  - *Embrace divine lighting, consider illuminating thy guitar with radiant brilliance.*
  - *Consider the steadfast guitar angle for quality. It is not commanded, but a wise path to follow.*

**Contributing**

If you're a developer who knows Python:
- I have encountered some difficulties in creating an executable. Initially, I attempted to build it as a Docker image, but I abandoned that approach due to the complexity of sharing X Server access with regular users. Furthermore, options like Pyinstaller are not suitable for generating a Windows build since I primarily use Linux.
- My code could benefit from further cleaning and refinement.
- I am eagerly seeking a modern graphical user interface (GUI) to enhance the user experience.

If you're a developer who knows C/C++ and fulfill the requirements outlined in the [YOLOv8-TensorRT-CPP repository](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP):
- While a C/C++ integration may not be immediately necessary at this early stage, the Python version of the program performs poorly on my machine, achieving a meager 10-12 frames per second (FPS). Introducing a C/C++ integration could potentially resolve this issue.

If you're a developer who wishes to port this program to another language such as Java or JavaScript:
- Your altruism is truly commendable! I've had a strong desire to produce an Android and web version of the program for a long time now, but my experience with Java is extremely limited, and I possess no knowledge of JavaScript. As a result, you would need to take the lead in this endeavor.

### Updates
**No new updates...**
You'll see them here described once I implement new features. Currently, my hands are full with school.

## Contact
Please feel free to contact me in **English**, **German**, or **Turkish**:
- Website: [https://di0n-0.github.io/](https://di0n-0.github.io/)
- Email: [di0nderentwickler@gmail.com](mailto:di0nderentwickler@gmail.com)
