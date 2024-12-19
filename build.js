const fs = require('fs');
const { JSDOM } = require('jsdom');
const postcss = require('postcss');
const cssnano = require('cssnano');
const PurgeCSS = require('purgecss').PurgeCSS;
const critical = require('critical');

// Load the HTML file
const html = fs.readFileSync('src/main.html', 'utf8');
const dom = new JSDOM(html);
const document = dom.window.document;

// Define the labelRefs function
function labelRefs() {
  let refs = document.querySelectorAll('a[href^="#fig"], a[href^="#table"]');
  let figures = document.querySelectorAll('figure');
  figures.forEach((figure, index) => {
    let caption = figure.querySelector('figcaption');
    let innerHTMLWithoutSpan = caption.innerHTML.replace(/<span>.*?<\/span>/, '');
    caption.innerHTML = `<span>Figure ${index + 1}</span>${innerHTMLWithoutSpan}`;
    refs.forEach(ref => {
      if (ref.getAttribute('href').substring(1) == figure.id) {
        ref.textContent = `${index + 1}`;
        ref.classList.add('fig-ref');
      }
    });
  });
//   let tables = document.querySelectorAll('table');
//   tables.forEach((table, index) => {
//     let caption = table.querySelector('caption');
//     caption.innerHTML = `Table ${index + 1}: ${caption.innerHTML}`;
//     refs.forEach(ref => {
//       if (ref.getAttribute('href').substring(1) == table.id) {
//         ref.textContent = `${index + 1}`;
//       }
//     });
//   });
}

// Execute the labelRefs function
labelRefs();

// Save the modified HTML back to the file
fs.writeFileSync('index.html', dom.serialize());

// // Minify the CSS
// const css = fs.readFileSync('src/css/main.css', 'utf8');
// postcss([cssnano])
//   .process(css, { from: 'assets/css/reduced.css', to: 'assets/css/reduced.min.css' })
//   .then(result => {
//     fs.writeFileSync('assets/css/reduced.min.css', result.css);
//     if (result.map) {
//       fs.writeFileSync('assets/css/reduced.min.css.map', result.map.toString());
//     }
//   });

// Iterate through all iframes and update data-src attribute
const iframes = document.querySelectorAll('iframe[data-src]');
iframes.forEach(iframe => {
  const dataSrc = iframe.getAttribute('data-src');
  const newSrc = dataSrc.replace('src/', 'dist/');
  iframe.setAttribute('data-src', newSrc);

  // Copy the file from src to dist directory
  const srcPath = dataSrc;
  const distPath = newSrc;
  fs.copyFileSync(srcPath, distPath);
});


// Purge unused CSS
const purgeCSSResults = new PurgeCSS().purge({
  content: ['index.html'],
  css: ['src/css/main.css'],
  safelist: ['canvas', '.fig-ref'],
});

purgeCSSResults.then(result => {
  const purgedCSS = result[0].css;

  // Minify the CSS
  postcss([cssnano])
    .process(purgedCSS, { from: 'src/css/main.css', to: 'dist/css/style.min.css' })
    .then(result => {
      fs.writeFileSync('dist/css/style.min.css', result.css);
      if (result.map) {
        fs.writeFileSync('dist/css/style.min.css.map', result.map.toString());
      }

      // Inline critical CSS
      critical.generate({
        inline: true,
        base: '.',
        src: 'index.html',
        css: ['dist/css/style.min.css'],
        target: {
          html: 'index.html',
          css: 'dist/css/uncritical.css'
        },
        extract: true
      });
    });
});