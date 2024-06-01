SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_07-26-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
UNION ALL
SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_08-02-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
UNION ALL
SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_08-09-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
UNION ALL
SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_08-16-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
UNION ALL
SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_08-23-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
UNION ALL
SELECT TimeStamp, DeviceId, EventId, Parameter
FROM [dbo].[ASC_PhasePed_Events_08-30-2021]
WHERE DeviceId = 217
	AND EventId IN (1, 8, 10)
	AND Parameter IN (2, 4, 6, 8)
	AND TimeStamp BETWEEN '2021-08-01' AND '2021-08-31'
