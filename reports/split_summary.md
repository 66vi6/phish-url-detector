# Split summary

| Split | Rows | Phishing | Benign | Phishing share | Phish date range |
|---|---|---|---|---|---|
| train | 28,776 | 28,776 | 0 | 100.0% | 2013-05-28 15:59 to 2026-01-24 19:01 |
| val | 745 | 745 | 0 | 100.0% | 2026-01-25 10:39 to 2026-05-16 02:23 |
| test | 3,493 | 3,493 | 0 | 100.0% | 2026-05-16 15:45 to 2026-11-18 09:04 |

Rows dropped to keep domains in one split only: 8,096 (19.7% of cleaned data)

Dropped by label:

label
1    8096

Top domains causing drops:

reg_domain
qrco.de          1743
q-r.to           1304
ead.me           1124
weebly.com        427
vercel.app        354
wixstudio.com     313
google.com        264
pages.dev         260
wl.co             189
framer.app        147
