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
        },
        {
          enable: true,
          selectors: ".bottom-bouncing",
          mode: "bounce",
          type: "rectangle"
        },
        {
          enable: true,
          selectors: ".right-bouncing",
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
        distance: 115,
        width: 1,
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
        value: 18
      },
      shape: {
          type: "image",
          options: {
            image: [
              {
                src: "./assets/img/beer_green.png",
                width: 100,
                height: 100
              },
              {
                src: "./assets/img/beer_brown.png",
                width: 100,
                height: 100
              }
          ]
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

(async () => {
  await loadSlim(tsParticles);
  await tsParticles.load(particles_img_config).then((container) => {
  });  
})();
