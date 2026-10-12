# Vendored front-end libraries

Served by main.py at /static/vendor/. No CDN at runtime: these are the whole
dependency. marked + DOMPurify load with the page; KaTeX is fetched only when a
message contains math (see ensureKatex in index.html). KaTeX's CSS also lists
.woff/.ttf fallbacks; only .woff2 is vendored, which every current browser uses.

To update: download the new tarball from registry.npmjs.org, check it against
the registry's dist.integrity (sha512), copy the same files, update this table.

| Package | Version | npm tarball integrity |
|---|---|---|
| marked | 15.0.6 | `sha512-Y07CUOE+HQXbVDCGl3LXggqJDbXDP2pArc2C1N1RRMN0ONiShoSsIInMd5Gsxupe7fKLpgimTV+HOJ9r7bA+pg==` |
| dompurify | 3.2.4 | `sha512-ysFSFEDVduQpyhzAob/kkuJjf5zWkZD8/A9ywSp1byueyuCfHamrCBa14/Oc2iiB0e51B+NpxSl5gmzn+Ms/mg==` |
| katex | 0.16.21 | `sha512-XvqR7FgOHtWupfMiigNzmh+MgUVmDGU2kXZm899ZkPfcuoPuFxyHmXsgATDpFZDAXCI8tvinaVcDo8PIIJSo4A==` |

Files (sha256):

```
1b02e03c3fb4f87d476c128f0eb9def1f5a1709d28b180465228bd41574623b7  ./dompurify/LICENSE
8eb41b658831fab175fad9bcd00fcb2d84e0ed3a25a55053d4ecd4444b8b43a0  ./dompurify/purify.min.js
0cdd387c9590a1a9f9794560022dbb59654a7d86f187aa0c81495ad42d3a7308  ./katex/fonts/KaTeX_AMS-Regular.woff2
de7701e42cf1f4cf0b766c03fb27977207eee2f4fd5d76fa82188406da43ea4c  ./katex/fonts/KaTeX_Caligraphic-Bold.woff2
5d53e70ad607c2352162dec9e0923fb54ecdafaccbf604cd8dcf7d00facb989b  ./katex/fonts/KaTeX_Caligraphic-Regular.woff2
74444efd593c005e3f4573b44524704c0af0a937fe911cca9e94068d0d140d3f  ./katex/fonts/KaTeX_Fraktur-Bold.woff2
51814d270d06ff0255dba0799994fa4d8c84d11f09951d47595f4abb1f3602dc  ./katex/fonts/KaTeX_Fraktur-Regular.woff2
99cd42a3c072d918f2f44984a807cf7aa16e13545fd0875fc07c6c65f99e715b  ./katex/fonts/KaTeX_Main-BoldItalic.woff2
0f60d1b897938ec918c8ce073092411baf9438f6739465693ff18b0f9d20b021  ./katex/fonts/KaTeX_Main-Bold.woff2
97479ca6cce906abc961ecac96faa5f9ca2e61b8e7670d475826bcdee9a7c267  ./katex/fonts/KaTeX_Main-Italic.woff2
c2342cd8b869e01752a9321dc17213fc40d4d04c79688c1d43f2cf316abd7866  ./katex/fonts/KaTeX_Main-Regular.woff2
dc47344dbb6cb5b655c8460d561f4df5f501b90c804ad3c6cec65fe322351ab1  ./katex/fonts/KaTeX_Math-BoldItalic.woff2
7af58c5ec8f132a2ddde9027c6d7814decce4d3b822a11192a42a20e2e973264  ./katex/fonts/KaTeX_Math-Italic.woff2
e99ae51144bf1232efcc1bfe5add36262c6866b0faab24fa75740e1b98577a62  ./katex/fonts/KaTeX_SansSerif-Bold.woff2
00b26ac825e2095056396e0553b8ac26d3f8ad158c3826e28b4c45b385c4714a  ./katex/fonts/KaTeX_SansSerif-Italic.woff2
68e8c73ef42afd3ccec58bf0fba302cce448938e7fc020a5e31f8a952eee1342  ./katex/fonts/KaTeX_SansSerif-Regular.woff2
036d4e95149b69ff9bcc0cd55771efeb25ffa3947293e69acd78d5ac328c684b  ./katex/fonts/KaTeX_Script-Regular.woff2
6b47c40166b6dbe21a5dfca7718413f2147fd2399be1ba605d8ad39cedf25dfe  ./katex/fonts/KaTeX_Size1-Regular.woff2
d04c54219f9eaec6d4d4fd42dfb28785975a4794d6b2fc71e566b9cd6db842dd  ./katex/fonts/KaTeX_Size2-Regular.woff2
73d591271b1604960cb10bb90fee021670af7297017e0e98480b332d11f51995  ./katex/fonts/KaTeX_Size3-Regular.woff2
a4af7d414440a1c1790825cfb700cf9cf43b0f2c4b04f0ebc523011ad9853ec0  ./katex/fonts/KaTeX_Size4-Regular.woff2
71d517d67827787cfabdf186914cc3358eda539e37931941f2b2fd4a21f68c0b  ./katex/fonts/KaTeX_Typewriter-Regular.woff2
f787891b550d554c214aa8902f39ac46df2dbd48fdec500a2040a5dce1e8ab58  ./katex/katex.min.css
863811e2baa0849c77bc92d26d44f4d0a4843c2a8ab52c462017dea57316e4d8  ./katex/katex.min.js
766ccc1f306c885aa45542a9846bbd0a505b27a0374f146778171c2254ce18e3  ./katex/LICENSE
8e3a3f82f59a60958f56ca08f445647c32a4733dc7ca6c2c46f6eb898471ab9c  ./marked/LICENSE.md
8b10a6173d524fc6c97bc9589ba8383e0212d5c0d617e1519e5f4e6a0c9f224f  ./marked/marked.min.js
```
