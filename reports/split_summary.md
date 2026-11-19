# Split summary

| Split | Rows | Phishing | Benign | Phishing share | Phish date range |
|---|---|---|---|---|---|
| train | 193,278 | 93,492 | 99,786 | 48.4% | 2024-07-02 00:00 to 2025-08-19 00:00 |
| val | 32,517 | 11,738 | 20,779 | 36.1% | 2025-08-20 00:00 to 2025-10-26 00:00 |
| test | 32,172 | 11,058 | 21,114 | 34.4% | 2025-10-27 00:00 to 2025-11-06 00:00 |

Rows dropped to keep domains in one split only: 18,184 (6.6% of cleaned data)

Dropped by label:

label
0      916
1    17268

Top domains causing drops:

reg_domain
pages.dev        4681
vercel.app        542
instagram.com     397
blogspot.com      310
netlify.app       271
vh.net.pl         186
github.io         177
sa.com            171
blogspot.be       170
blogspot.ae       154
