Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
GET      /health    3172     5(0.16%) |   8492     244   56463   7900 |    0.14        0.00
POST     /predict   15733    60(0.38%) |  67211      14  479127  57000 |    0.68        0.00
GET      /uptime    3200     7(0.22%) |  10541     589   69229   9900 |    0.14        0.00
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated   22105    72(0.33%) |  50581      14  479127  42000 |    0.96        0.00

Response time percentiles (approximated)
Type     Name      50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------||--------|------|------|------|------|------|------|------|------|------|------|------
GET      /health     7900   9200  10000  11000  13000  15000  20000  26000  44000  56000  56000   3172
POST     /predict    57000  77000  89000  96000 116000 132000 162000 199000 293000 381000 479000  15733
GET      /uptime     9900  11000  12000  13000  15000  17000  24000  30000  42000  69000  69000   3200
--------||--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated    42000  60000  75000  85000 107000 124000 148000 178000 290000 365000 479000  22105

Error report
# occurrences      Error                                                    
------------------|-------------------------------------------------------------
59                 POST /predict: RemoteDisconnected('Remote end closed connection without response')
7                  GET /uptime: RemoteDisconnected('Remote end closed connection without response')
5                  GET /health: RemoteDisconnected('Remote end closed connection without response')
1                  POST /predict: ConnectionAbortedError(10053, 'An established connection was aborted by the software in your host machine', None, 10053, None)
------------------|-------------------------------------------------------------
