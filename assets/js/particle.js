// import { loadPoissonDiscPlugin } from "https://cdn.jsdelivr.net/npm/@tsparticles/plugin-poisson-disc@3.7.1/+esm";

let particles_config = {
    id: "tsparticles",
    options: {
      fullScreen: {
        enable: false
      },
      fpsLimit: 60,
      interactivity: {
        detectsOn: "canvas",
        events: {
          onDiv: [{
            enable: true,
            selectors: ".hero-title",
            mode: "bounce",
            type: "rectangle"
          },
          {
            enable: true,
            selectors: "#bubble-circle-div",
            mode: "bounce",
            type: "circle"
          },],
          onClick: {
            enable: true,
            mode: "repulse"
          },
          onHover: {
            enable: true,
            mode: "repulse"
          },
          resize: true
        },
      },
      // poisson: {
      //   enable: true
      // },
      particles: {
        number: {
          value: 350,
          density: {
            enable: true,
          }
        },
        links: {
          enable: true,
          distance: 100,
          width: 2,
          color: {
            value: "#ff00ff"
          }
        },
        move: {
          enable: true,
          speed: 2,
          outModes: {
            default: "bounce"
          }
        },
        size: {
          value: 20
        },
        shape: {
            type: "emoji",
            options: {
              emoji: {
                value: ["🍺", "🍂", "🍃", "❄️", "☀️", "⏱️", "🍺", "🍺", "🍺"]
              }
            },
        },
      },
      themes: [
        {
          name: "light",
          default: {
            value: true,
            mode: "light",
            auto: true
          },
          options: {
            particles: {
              links: {
                color: {
                  value: "#212529"
                }
              }
            }
          }
        },
        {
          name: "dark",
          default: {
            value: true,
            mode: "dark",
            auto: true
          },
          options: {
            particles: {
              links: {
                color: {
                  value: "#ffffff"
                }
              }
            }
          }
        },
      ],
    }
};

let particles_img_config = {
  id: "tsparticles",
  options: {
    fullScreen: {
      enable: false
    },
    poisson: {
      enable: true
    },
    fpsLimit: 60,
    interactivity: {
      detectsOn: "canvas",
      events: {
        onDiv: [{
          enable: true,
          selectors: ".hero-title",
          mode: "bounce",
          type: "rectangle"
        },
        {
          enable: true,
          selectors: ".round-div-left",
          mode: "bounce",
          type: "circle"
        },
        {
          enable: true,
          selectors: ".round-div-right",
          mode: "bounce",
          type: "circle"
        },
        {
          enable: true,
          selectors: ".corner",
          mode: "bounce",
          type: "rectangle"
        }
      ],
        onClick: {
          enable: true,
          mode: "repulse"
        },
        onHover: {
          enable: true,
          mode: "repulse"
        },
        resize: true
      },
    },
    particles: {
      number: {
        value: 200,
        density: {
          enable: true,
        }
      },
      links: {
        enable: true,
        distance: 120,
        width: 1.5,
        color: {
          value: "#ff00ff"
        }
      },
      move: {
        enable: true,
        speed: 2,
        outModes: {
          default: "bounce"
        }
      },
      size: {
        value: 10
      },
      shape: {
          type: "image",
          options: {
            image: [
              {
                // src: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Pixel_art_Wizard_Portrait.svg/1920px-Pixel_art_Wizard_Portrait.svg.png",
                src: "./assets/img/beer_test2.png",
                width: 100,
                height: 200
              },
              {
                src: "./assets/img/beer_test3.png",
                width: 100,
                height: 200
              }
          ]
          },
      },
      // collisions: {
      //   enable: true,
      //   mode: "bounce"
      // }
    },
    // poisson: {
    //   enable: true
    // },
    themes: [
      {
        name: "light",
        default: {
          value: true,
          mode: "light",
          auto: true
        },
        options: {
          particles: {
            links: {
              color: {
                value: "#212529"
              }
            }
          }
        }
      },
      {
        name: "dark",
        default: {
          value: true,
          mode: "dark",
          auto: true
        },
        options: {
          particles: {
            links: {
              color: {
                value: "#ffffff"
              }
            }
          }
        }
      },
    ],
  }
};



// particles_config.options.particles.links.color.value = "#0000ff";



(async () => {
  await loadFull(tsParticles);
  // await loadAll(tsParticles);
  // await loadPoissonDiscPlugin(tsParticles);
  await tsParticles.load(particles_img_config).then((container) => {
    // container.loadTheme("dark");
  });  
  await tsParticles.refresh();
})();

// loadParticles(tsParticles);