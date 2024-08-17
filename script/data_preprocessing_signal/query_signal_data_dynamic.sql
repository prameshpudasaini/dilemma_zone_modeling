-- declare variable '@sql' to hold dynamically constructed SQL query
DECLARE @sql NVARCHAR(MAX);

-- initialize SQL variable
SET @sql = '';

-- list of table names
-- declare variable 'tables' to temporarily store list of tables with 'TableName' as column
DECLARE @tables TABLE (TableName NVARCHAR(128));

-- insert table names into @tables
INSERT INTO @tables (TableName) VALUES
('[dbo].[ASC_PhasePed_Events_07-26-2021]'),
('[dbo].[ASC_PhasePed_Events_08-02-2021]'),
('[dbo].[ASC_PhasePed_Events_08-09-2021]'),
('[dbo].[ASC_PhasePed_Events_08-16-2021]'),
('[dbo].[ASC_PhasePed_Events_08-23-2021]'),
('[dbo].[ASC_PhasePed_Events_08-30-2021]');

-- build dynamic SQL query by iterating through each table name
SELECT @sql = @sql + '
	SELECT TimeStamp, DeviceId, EventId, Parameter
	FROM ' + TableName + '
	WHERE DeviceId = 217
		AND EventId IN (1, 8, 10)
		AND Parameter IN (2, 4, 6, 8)
		AND TimeStamp BETWEEN ''2021-08-01'' AND ''2021-08-31''
	UNION ALL'
FROM @tables;

-- remove the trailing UNION ALL from the final SQL string
SET @sql = LEFT(@sql, LEN(@sql) - LEN(' UNION ALL'));

-- execute dynamic SQL
EXEC sp_executesql @sql;