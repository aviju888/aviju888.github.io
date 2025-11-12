# Portfolio Transformation Checklist

Quick reference checklist for transforming the portfolio. Use this alongside the detailed guide.

## 🔴 CRITICAL CHANGES (Do First)

### Main Site (`index.html`)
- [ ] Line 6: Change title from "CS180 Projects" to "Computer Vision Portfolio" or "Adriel Vijuan"
- [ ] Line 15: Change header from "CS180: Projects" to "Computer Vision Portfolio" or your name
- [ ] Line 25: "Project #1" → "Historical Photo Colorization"
- [ ] Line 30: "Project #2" → "Image Filtering & Frequency Analysis"
- [ ] Line 35: "Project #3" → "Face Morphing & Shape Averaging"
- [ ] Line 40: "Project #4" → "Panoramic Image Stitching"
- [ ] Line 45: "Project #5" → "Diffusion Models"
- [ ] Line 50: "Final Project (pt.1)" → "High Dynamic Range Imaging"
- [ ] Line 55: "Final Project (pt.2)" → "Light Field Camera & Depth Estimation"
- [ ] Line 63: Remove "for CS180 - 2024", change to "© 2024 Adriel Vijuan" or remove

### All Project Pages
- [ ] Remove "CS180:" from all `<title>` tags
- [ ] Remove "CS180:" from all `<h1>` headers
- [ ] Change author attribution from `<h3>` to subtle text or footer
- [ ] Update page titles to descriptive names

---

## 🟡 IMPORTANT CHANGES (Do Second)

### Language Replacements (Search & Replace Across All Files)
- [ ] "provided data folder" → "dataset"
- [ ] "provided data" → "dataset"
- [ ] "the spec" → [delete]
- [ ] "discussed in lecture" → "based on established methods" or [delete]
- [ ] "this project focuses on" → "This work implements" or "I developed"
- [ ] "the project involves" → "The implementation includes"
- [ ] "I used the provided" → "I implemented" or "I developed"

### Section Headers
- [ ] "Project Overview" → "Overview" or "Introduction"
- [ ] "Approach Breakdown" → "Methodology" or "Implementation"
- [ ] "Bells and Whistles" → "Additional Features" or "Enhancements"
- [ ] "Commentary" → "Discussion" or "Analysis"
- [ ] "Step-by-Step Process" → "Methodology" or "Implementation Details"
- [ ] "Part A" / "Part B" → Descriptive section names

### Project-Specific Fixes

#### Project 1 (`projects/proj1/web/index.html`)
- [ ] Line 6: Title → "Historical Photo Colorization - Adriel Vijuan"
- [ ] Line 15: Header → "Historical Photo Colorization"
- [ ] Line 25: "Project Overview" → "Overview"
- [ ] Line 52: "Approach Breakdown" → "Methodology"
- [ ] Line 167: "Bells and Whistles" → "Additional Features"
- [ ] Line 242: "Commentary" → "Discussion"
- [ ] Remove "provided data folder" references
- [ ] Improve image captions (less technical, more descriptive)

#### Project 3 (`projects/proj3/web/index.html`)
- [ ] Line 6: Title → "Face Morphing - Adriel Vijuan"
- [ ] Line 15: Header → "Face Morphing"
- [ ] Remove personal Photoshop anecdote (lines 159-164)
- [ ] "I used the provided Correspondence tool" → "I implemented correspondence point selection"
- [ ] "Mean Face" → "Population Face Averaging"

#### Project 4 (`projects/proj4/web/index.html`)
- [ ] Line 6: Title → "Panoramic Image Stitching - Adriel Vijuan"
- [ ] Line 15: Header → "Panoramic Image Stitching"
- [ ] "Part A" → "Image Warping & Alignment"
- [ ] "Part B" → "Automated Feature Detection" (if applicable)
- [ ] "Step-by-Step Process" → "Methodology"

#### Project 6 (`projects/proj6/web/index.html`)
- [ ] Line 6: Title → "High Dynamic Range Imaging - Adriel Vijuan"
- [ ] Line 15: Header → "High Dynamic Range Imaging"
- [ ] Remove "Exploring Part A and Part B" from header
- [ ] "Project Goals" → "Objectives" or merge into Introduction

---

## 🟢 ENHANCEMENTS (Do Third - Optional but Recommended)

### Code Files
- [ ] `script.js` line 2: Update comment
- [ ] `style.css` lines 1-3: Update comment
- [ ] Check Python files for academic references

### Meta Tags (All HTML Files)
- [ ] Add `<meta name="description">` to all pages
- [ ] Add Open Graph tags for social sharing
- [ ] Add keywords meta tags

### Visual Improvements
- [ ] Make author attribution subtle (not prominent `<h3>`)
- [ ] Improve image captions (descriptive primary, technical secondary)
- [ ] Add technology tags/badges to projects
- [ ] Add "View Code" links if you have repositories

### Navigation
- [ ] Add breadcrumbs
- [ ] Add consistent "Back to Portfolio" links
- [ ] Ensure all internal links work after changes

---

## 📝 QUICK SEARCH & REPLACE LIST

Run these searches across all HTML files:

1. **"CS180"** → [delete or replace with nothing]
2. **"Project #"** → [replace with descriptive name]
3. **"provided data"** → "dataset"
4. **"the spec"** → [delete]
5. **"discussed in lecture"** → [delete or "based on established methods"]
6. **"this project"** → "this work" or "this implementation"
7. **"Created by.*for CS180"** → "© 2024 Adriel Vijuan"

---

## ✅ VERIFICATION CHECKLIST

After making changes, verify:

- [ ] No "CS180" appears anywhere visible to users
- [ ] No "Project #X" appears (use descriptive names)
- [ ] All internal links still work
- [ ] Page titles are descriptive and professional
- [ ] No academic assignment language remains
- [ ] Author attribution is subtle, not prominent
- [ ] Footer doesn't mention course
- [ ] All images load correctly
- [ ] Site is consistent across all pages

---

## 🎯 PRIORITY ORDER

1. **Remove all course references** (CS180, course numbers)
2. **Rename projects** (Project #1 → Descriptive names)
3. **Update language** (academic → professional)
4. **Fix section headers** (Bells and Whistles → Enhancements)
5. **Remove personal anecdotes** (Photoshop story)
6. **Add professional touches** (meta tags, better navigation)

---

## 📌 NOTES

- Keep all technical content - only change presentation
- Don't exaggerate or lie about accomplishments
- Focus on what you built, not that it was an assignment
- Test thoroughly after making changes
- Consider version control (git) before starting

