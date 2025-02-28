/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html","./src/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: [
      {
        marble: {
          "primary": "#c4a47c",
          "secondary": "#6b7280",
          "accent": "#f3f4f6",
          "neutral": "#2b3440",
          "base-100": "#ffffff",
          "dark": "#000000",
        },
    
       } // or ["light", "dark", "cupcake", ...] to include multiple themes
      ],
}
}

