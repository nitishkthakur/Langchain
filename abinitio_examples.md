# Ab-Initio (.mp) to SQL Conversion Examples

This document provides 5 paired examples of Ab-Initio graph (.mp) files and their equivalent SQL translations, arranged in increasing order of complexity. Each pair demonstrates how Ab-Initio components and data flows translate to SQL queries.

---

## Example 1: Simple SELECT with Filter (EASY)

### Ab-Initio (.mp) File: `example1_simple_filter.mp`

```
// Component: Input File
component: input_file_customers
type: INPUT_FILE
out: customers_stream
metadata:
  - customer_id: integer
  - customer_name: string
  - age: integer
  - country: string

// Component: Filter
component: filter_adult_customers
type: FILTER
in1: customers_stream
out: filtered_customers
notation: age >= 18

// Component: Output File
component: output_file_result
type: OUTPUT_FILE
in1: filtered_customers
```

### SQL File: `example1_simple_filter.sql`

```sql
-- Simple SELECT with WHERE clause
SELECT
    customer_id,
    customer_name,
    age,
    country
FROM customers
WHERE age >= 18;
```

**Mapping Notes:**
- INPUT_FILE → FROM clause
- FILTER component → WHERE clause
- OUTPUT_FILE → Result set

---

## Example 2: SELECT with Transformation (EASY)

### Ab-Initio (.mp) File: `example2_simple_transform.mp`

```
// Component: Input File
component: input_file_orders
type: INPUT_FILE
out: orders_stream
metadata:
  - order_id: integer
  - product_name: string
  - quantity: integer
  - unit_price: decimal(10,2)

// Component: Reformat
component: reformat_calculate_total
type: REFORMAT
in1: orders_stream
out: transformed_orders
notation:
  order_id = order_id
  product_name = product_name
  quantity = quantity
  unit_price = unit_price
  total_amount = quantity * unit_price

// Component: Filter
component: filter_high_value
type: FILTER
in1: transformed_orders
out: high_value_orders
notation: total_amount > 1000

// Component: Output File
component: output_file_result
type: OUTPUT_FILE
in1: high_value_orders
```

### SQL File: `example2_simple_transform.sql`

```sql
-- SELECT with calculated column and filter
SELECT
    order_id,
    product_name,
    quantity,
    unit_price,
    quantity * unit_price AS total_amount
FROM orders
WHERE quantity * unit_price > 1000;
```

**Mapping Notes:**
- REFORMAT component → Calculated columns in SELECT
- notation fields → Column expressions
- Multiple FILTER → Combined WHERE conditions

---

## Example 3: JOIN with Aggregation (INTERMEDIATE)

### Ab-Initio (.mp) File: `example3_join_aggregate.mp`

```
// Component: Input File - Employees
component: input_file_employees
type: INPUT_FILE
out: employees_stream
metadata:
  - employee_id: integer
  - employee_name: string
  - department_id: integer
  - salary: decimal(10,2)
  - hire_date: date

// Component: Input File - Departments
component: input_file_departments
type: INPUT_FILE
out: departments_stream
metadata:
  - department_id: integer
  - department_name: string
  - location: string

// Component: Join
component: join_emp_dept
type: JOIN
in1: employees_stream
in2: departments_stream
out: joined_stream
join_type: INNER
join_key_in1: department_id
join_key_in2: department_id

// Component: Reformat
component: reformat_fields
type: REFORMAT
in1: joined_stream
out: formatted_stream
notation:
  employee_name = employee_name
  department_name = department_name
  salary = salary
  location = location
  years_of_service = datediff(current_date(), hire_date) / 365

// Component: Filter
component: filter_high_salary
type: FILTER
in1: formatted_stream
out: filtered_stream
notation: salary > 50000 AND years_of_service > 2

// Component: Output File
component: output_file_result
type: OUTPUT_FILE
in1: filtered_stream
```

### SQL File: `example3_join_aggregate.sql`

```sql
-- JOIN with transformation and filtering
SELECT
    e.employee_name,
    d.department_name,
    e.salary,
    d.location,
    DATEDIFF(CURRENT_DATE, e.hire_date) / 365 AS years_of_service
FROM employees e
INNER JOIN departments d
    ON e.department_id = d.department_id
WHERE e.salary > 50000
    AND DATEDIFF(CURRENT_DATE, e.hire_date) / 365 > 2;
```

**Mapping Notes:**
- JOIN component → JOIN clause with ON condition
- join_key_in1/in2 → JOIN keys
- REFORMAT after JOIN → SELECT expressions
- Chained transformations → Sequential query logic

---

## Example 4: Multiple JOINs with CTEs (COMPLEX)

### Ab-Initio (.mp) File: `example4_complex_multi_join.mp`

```
// Component: Input File - Sales
component: input_file_sales
type: INPUT_FILE
out: sales_stream
metadata:
  - sale_id: integer
  - product_id: integer
  - customer_id: integer
  - salesperson_id: integer
  - sale_amount: decimal(10,2)
  - sale_date: date
  - region_id: integer

// Component: Input File - Products
component: input_file_products
type: INPUT_FILE
out: products_stream
metadata:
  - product_id: integer
  - product_name: string
  - category: string
  - cost: decimal(10,2)

// Component: Input File - Customers
component: input_file_customers
type: INPUT_FILE
out: customers_stream
metadata:
  - customer_id: integer
  - customer_name: string
  - customer_type: string
  - credit_limit: decimal(10,2)

// Component: Input File - Salespersons
component: input_file_salespersons
type: INPUT_FILE
out: salespersons_stream
metadata:
  - salesperson_id: integer
  - salesperson_name: string
  - commission_rate: decimal(5,2)

// Component: Input File - Regions
component: input_file_regions
type: INPUT_FILE
out: regions_stream
metadata:
  - region_id: integer
  - region_name: string
  - region_manager: string

// Component: Join - Sales with Products
component: join_sales_products
type: JOIN
in1: sales_stream
in2: products_stream
out: sales_products_stream
join_type: INNER
join_key_in1: product_id
join_key_in2: product_id

// Component: Join - Add Customers
component: join_add_customers
type: JOIN
in1: sales_products_stream
in2: customers_stream
out: sales_prod_cust_stream
join_type: LEFT
join_key_in1: customer_id
join_key_in2: customer_id

// Component: Join - Add Salespersons
component: join_add_salespersons
type: JOIN
in1: sales_prod_cust_stream
in2: salespersons_stream
out: sales_all_stream
join_type: INNER
join_key_in1: salesperson_id
join_key_in2: salesperson_id

// Component: Join - Add Regions
component: join_add_regions
type: JOIN
in1: sales_all_stream
in2: regions_stream
out: complete_sales_stream
join_type: LEFT
join_key_in1: region_id
join_key_in2: region_id

// Component: Reformat - Calculate Metrics
component: reformat_metrics
type: REFORMAT
in1: complete_sales_stream
out: metrics_stream
notation:
  sale_id = sale_id
  product_name = product_name
  category = category
  customer_name = customer_name
  customer_type = customer_type
  salesperson_name = salesperson_name
  region_name = region_name
  sale_amount = sale_amount
  cost = cost
  profit = sale_amount - cost
  commission = sale_amount * commission_rate / 100
  profit_margin = ((sale_amount - cost) / sale_amount) * 100
  sale_year = year(sale_date)
  sale_quarter = quarter(sale_date)

// Component: Filter - Profitable Sales
component: filter_profitable
type: FILTER
in1: metrics_stream
out: filtered_metrics
notation: profit > 0 AND sale_year >= 2023

// Component: Aggregate - Summary by Category
component: aggregate_category
type: AGGREGATE
in1: filtered_metrics
out: category_summary
group_by: category, sale_year, sale_quarter
notation:
  category = category
  sale_year = sale_year
  sale_quarter = sale_quarter
  total_sales = sum(sale_amount)
  total_profit = sum(profit)
  total_commission = sum(commission)
  avg_profit_margin = avg(profit_margin)
  transaction_count = count(*)

// Component: Filter - Significant Categories
component: filter_significant
type: FILTER
in1: category_summary
out: significant_categories
notation: total_sales > 10000

// Component: Output File
component: output_file_result
type: OUTPUT_FILE
in1: significant_categories
```

### SQL File: `example4_complex_multi_join.sql`

```sql
-- Complex query with multiple CTEs and JOINs
WITH sales_enriched AS (
    -- CTE 1: Join all dimension tables to sales facts
    SELECT
        s.sale_id,
        p.product_name,
        p.category,
        c.customer_name,
        c.customer_type,
        sp.salesperson_name,
        r.region_name,
        s.sale_amount,
        p.cost,
        sp.commission_rate,
        s.sale_date
    FROM sales s
    INNER JOIN products p
        ON s.product_id = p.product_id
    LEFT JOIN customers c
        ON s.customer_id = c.customer_id
    INNER JOIN salespersons sp
        ON s.salesperson_id = sp.salesperson_id
    LEFT JOIN regions r
        ON s.region_id = r.region_id
),
sales_metrics AS (
    -- CTE 2: Calculate business metrics
    SELECT
        sale_id,
        product_name,
        category,
        customer_name,
        customer_type,
        salesperson_name,
        region_name,
        sale_amount,
        cost,
        sale_amount - cost AS profit,
        sale_amount * commission_rate / 100 AS commission,
        ((sale_amount - cost) / sale_amount) * 100 AS profit_margin,
        YEAR(sale_date) AS sale_year,
        QUARTER(sale_date) AS sale_quarter
    FROM sales_enriched
    WHERE sale_amount - cost > 0
        AND YEAR(sale_date) >= 2023
)
SELECT
    category,
    sale_year,
    sale_quarter,
    SUM(sale_amount) AS total_sales,
    SUM(profit) AS total_profit,
    SUM(commission) AS total_commission,
    AVG(profit_margin) AS avg_profit_margin,
    COUNT(*) AS transaction_count
FROM sales_metrics
GROUP BY category, sale_year, sale_quarter
HAVING SUM(sale_amount) > 10000
ORDER BY sale_year DESC, sale_quarter DESC, total_sales DESC;
```

**Mapping Notes:**
- Multiple INPUT_FILE → Multiple source tables
- Sequential JOIN components → Multiple JOIN clauses
- join_type parameter → INNER/LEFT JOIN types
- REFORMAT with calculations → CTE with computed columns
- AGGREGATE component → GROUP BY with aggregate functions
- group_by notation → GROUP BY clause
- Post-aggregate FILTER → HAVING clause

---

## Example 5: Advanced Multi-CTE with Complex Transformations (COMPLEX)

### Ab-Initio (.mp) File: `example5_advanced_multi_cte.mp`

```
// Component: Input File - Transactions
component: input_file_transactions
type: INPUT_FILE
out: transactions_stream
metadata:
  - transaction_id: integer
  - account_id: integer
  - transaction_date: datetime
  - transaction_type: string
  - amount: decimal(15,2)
  - merchant_id: integer
  - status: string

// Component: Input File - Accounts
component: input_file_accounts
type: INPUT_FILE
out: accounts_stream
metadata:
  - account_id: integer
  - customer_id: integer
  - account_type: string
  - opening_date: date
  - current_balance: decimal(15,2)
  - risk_score: integer

// Component: Input File - Customers
component: input_file_customers
type: INPUT_FILE
out: customers_stream
metadata:
  - customer_id: integer
  - customer_name: string
  - customer_segment: string
  - country: string
  - registration_date: date

// Component: Input File - Merchants
component: input_file_merchants
type: INPUT_FILE
out: merchants_stream
metadata:
  - merchant_id: integer
  - merchant_name: string
  - merchant_category: string
  - country: string

// Component: Input File - Exchange Rates
component: input_file_exchange_rates
type: INPUT_FILE
out: exchange_rates_stream
metadata:
  - rate_date: date
  - from_country: string
  - to_country: string
  - exchange_rate: decimal(10,6)

// Component: Filter - Recent Transactions
component: filter_recent_trans
type: FILTER
in1: transactions_stream
out: recent_trans_stream
notation: transaction_date >= add_months(current_date(), -6) AND status = 'COMPLETED'

// Component: Join - Transactions with Accounts
component: join_trans_accounts
type: JOIN
in1: recent_trans_stream
in2: accounts_stream
out: trans_accounts_stream
join_type: INNER
join_key_in1: account_id
join_key_in2: account_id

// Component: Join - Add Customers
component: join_add_customers
type: JOIN
in1: trans_accounts_stream
in2: customers_stream
out: trans_acc_cust_stream
join_type: INNER
join_key_in1: customer_id
join_key_in2: customer_id

// Component: Join - Add Merchants
component: join_add_merchants
type: JOIN
in1: trans_acc_cust_stream
in2: merchants_stream
out: trans_complete_stream
join_type: LEFT
join_key_in1: merchant_id
join_key_in2: merchant_id

// Component: Join - Add Exchange Rates
component: join_add_exchange
type: JOIN
in1: trans_complete_stream
in2: exchange_rates_stream
out: trans_with_rates_stream
join_type: LEFT
join_key_in1: country, date(transaction_date)
join_key_in2: from_country, rate_date
additional_condition: exchange_rates_stream.to_country = 'USA'

// Component: Reformat - Transaction Metrics
component: reformat_trans_metrics
type: REFORMAT
in1: trans_with_rates_stream
out: trans_metrics_stream
notation:
  transaction_id = transaction_id
  account_id = account_id
  customer_id = customer_id
  customer_name = customer_name
  customer_segment = customer_segment
  account_type = account_type
  merchant_name = coalesce(merchant_name, 'UNKNOWN')
  merchant_category = coalesce(merchant_category, 'OTHER')
  transaction_type = transaction_type
  transaction_date = transaction_date
  amount_local = amount
  amount_usd = amount * coalesce(exchange_rate, 1.0)
  risk_score = risk_score
  trans_month = date_format(transaction_date, 'YYYY-MM')
  trans_year = year(transaction_date)
  is_high_risk = if(risk_score > 70, 1, 0)
  is_large_transaction = if(amount > 5000, 1, 0)
  customer_tenure_days = datediff(transaction_date, registration_date)

// Component: Aggregate - Customer Monthly Summary
component: aggregate_customer_monthly
type: AGGREGATE
in1: trans_metrics_stream
out: customer_monthly_stream
group_by: customer_id, customer_name, customer_segment, trans_month, trans_year
notation:
  customer_id = customer_id
  customer_name = customer_name
  customer_segment = customer_segment
  trans_month = trans_month
  trans_year = trans_year
  total_transactions = count(*)
  total_amount_usd = sum(amount_usd)
  avg_transaction_usd = avg(amount_usd)
  max_transaction_usd = max(amount_usd)
  high_risk_trans_count = sum(is_high_risk)
  large_trans_count = sum(is_large_transaction)
  unique_merchants = count(distinct merchant_id)
  unique_categories = count(distinct merchant_category)

// Component: Reformat - Risk Indicators
component: reformat_risk_indicators
type: REFORMAT
in1: customer_monthly_stream
out: risk_indicators_stream
notation:
  customer_id = customer_id
  customer_name = customer_name
  customer_segment = customer_segment
  trans_month = trans_month
  trans_year = trans_year
  total_transactions = total_transactions
  total_amount_usd = total_amount_usd
  avg_transaction_usd = avg_transaction_usd
  max_transaction_usd = max_transaction_usd
  high_risk_trans_count = high_risk_trans_count
  large_trans_count = large_trans_count
  unique_merchants = unique_merchants
  unique_categories = unique_categories
  risk_ratio = high_risk_trans_count / total_transactions
  large_trans_ratio = large_trans_count / total_transactions
  risk_flag = if(risk_ratio > 0.3 OR large_trans_ratio > 0.5, 'HIGH_RISK', 'NORMAL')

// Component: Aggregate - Merchant Category Summary
component: aggregate_merchant_category
type: AGGREGATE
in1: trans_metrics_stream
out: merchant_category_stream
group_by: merchant_category, trans_year, trans_month
notation:
  merchant_category = merchant_category
  trans_year = trans_year
  trans_month = trans_month
  category_trans_count = count(*)
  category_total_usd = sum(amount_usd)
  category_avg_usd = avg(amount_usd)
  unique_customers = count(distinct customer_id)

// Component: Join - Combine Customer and Category Analysis
component: join_customer_category
type: JOIN
in1: risk_indicators_stream
in2: merchant_category_stream
out: combined_analysis_stream
join_type: INNER
join_key_in1: trans_month, trans_year
join_key_in2: trans_month, trans_year

// Component: Reformat - Final Enrichment
component: reformat_final
type: REFORMAT
in1: combined_analysis_stream
out: final_stream
notation:
  customer_id = customer_id
  customer_name = customer_name
  customer_segment = customer_segment
  trans_month = trans_month
  trans_year = trans_year
  total_transactions = total_transactions
  total_amount_usd = total_amount_usd
  avg_transaction_usd = avg_transaction_usd
  max_transaction_usd = max_transaction_usd
  risk_flag = risk_flag
  risk_ratio = risk_ratio
  large_trans_ratio = large_trans_ratio
  unique_merchants = unique_merchants
  merchant_category = merchant_category
  category_total_usd = category_total_usd
  category_share_pct = (total_amount_usd / category_total_usd) * 100

// Component: Filter - High Value Risk Customers
component: filter_high_value_risk
type: FILTER
in1: final_stream
out: filtered_final_stream
notation: total_amount_usd > 50000 AND (risk_flag = 'HIGH_RISK' OR max_transaction_usd > 10000)

// Component: Sort
component: sort_results
type: SORT
in1: filtered_final_stream
out: sorted_stream
sort_keys: trans_year DESC, trans_month DESC, total_amount_usd DESC

// Component: Output File
component: output_file_result
type: OUTPUT_FILE
in1: sorted_stream
```

### SQL File: `example5_advanced_multi_cte.sql`

```sql
-- Advanced multi-CTE query with complex business logic
WITH recent_transactions AS (
    -- CTE 1: Filter for recent completed transactions
    SELECT
        transaction_id,
        account_id,
        transaction_date,
        transaction_type,
        amount,
        merchant_id,
        status
    FROM transactions
    WHERE transaction_date >= DATE_ADD(CURRENT_DATE, INTERVAL -6 MONTH)
        AND status = 'COMPLETED'
),
enriched_transactions AS (
    -- CTE 2: Join all dimension tables
    SELECT
        t.transaction_id,
        t.account_id,
        a.customer_id,
        c.customer_name,
        c.customer_segment,
        a.account_type,
        COALESCE(m.merchant_name, 'UNKNOWN') AS merchant_name,
        COALESCE(m.merchant_category, 'OTHER') AS merchant_category,
        t.transaction_type,
        t.transaction_date,
        t.amount AS amount_local,
        t.amount * COALESCE(er.exchange_rate, 1.0) AS amount_usd,
        a.risk_score,
        DATE_FORMAT(t.transaction_date, '%Y-%m') AS trans_month,
        YEAR(t.transaction_date) AS trans_year,
        CASE WHEN a.risk_score > 70 THEN 1 ELSE 0 END AS is_high_risk,
        CASE WHEN t.amount > 5000 THEN 1 ELSE 0 END AS is_large_transaction,
        DATEDIFF(t.transaction_date, c.registration_date) AS customer_tenure_days,
        t.merchant_id
    FROM recent_transactions t
    INNER JOIN accounts a
        ON t.account_id = a.account_id
    INNER JOIN customers c
        ON a.customer_id = c.customer_id
    LEFT JOIN merchants m
        ON t.merchant_id = m.merchant_id
    LEFT JOIN exchange_rates er
        ON c.country = er.from_country
        AND DATE(t.transaction_date) = er.rate_date
        AND er.to_country = 'USA'
),
customer_monthly_summary AS (
    -- CTE 3: Aggregate metrics by customer and month
    SELECT
        customer_id,
        customer_name,
        customer_segment,
        trans_month,
        trans_year,
        COUNT(*) AS total_transactions,
        SUM(amount_usd) AS total_amount_usd,
        AVG(amount_usd) AS avg_transaction_usd,
        MAX(amount_usd) AS max_transaction_usd,
        SUM(is_high_risk) AS high_risk_trans_count,
        SUM(is_large_transaction) AS large_trans_count,
        COUNT(DISTINCT merchant_id) AS unique_merchants,
        COUNT(DISTINCT merchant_category) AS unique_categories
    FROM enriched_transactions
    GROUP BY customer_id, customer_name, customer_segment, trans_month, trans_year
),
customer_risk_indicators AS (
    -- CTE 4: Calculate risk indicators
    SELECT
        customer_id,
        customer_name,
        customer_segment,
        trans_month,
        trans_year,
        total_transactions,
        total_amount_usd,
        avg_transaction_usd,
        max_transaction_usd,
        high_risk_trans_count,
        large_trans_count,
        unique_merchants,
        unique_categories,
        high_risk_trans_count / total_transactions AS risk_ratio,
        large_trans_count / total_transactions AS large_trans_ratio,
        CASE
            WHEN high_risk_trans_count / total_transactions > 0.3
                OR large_trans_count / total_transactions > 0.5
            THEN 'HIGH_RISK'
            ELSE 'NORMAL'
        END AS risk_flag
    FROM customer_monthly_summary
),
merchant_category_summary AS (
    -- CTE 5: Aggregate by merchant category
    SELECT
        merchant_category,
        trans_year,
        trans_month,
        COUNT(*) AS category_trans_count,
        SUM(amount_usd) AS category_total_usd,
        AVG(amount_usd) AS category_avg_usd,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM enriched_transactions
    GROUP BY merchant_category, trans_year, trans_month
)
SELECT
    cri.customer_id,
    cri.customer_name,
    cri.customer_segment,
    cri.trans_month,
    cri.trans_year,
    cri.total_transactions,
    cri.total_amount_usd,
    cri.avg_transaction_usd,
    cri.max_transaction_usd,
    cri.risk_flag,
    cri.risk_ratio,
    cri.large_trans_ratio,
    cri.unique_merchants,
    mcs.merchant_category,
    mcs.category_total_usd,
    (cri.total_amount_usd / mcs.category_total_usd) * 100 AS category_share_pct
FROM customer_risk_indicators cri
INNER JOIN merchant_category_summary mcs
    ON cri.trans_month = mcs.trans_month
    AND cri.trans_year = mcs.trans_year
WHERE cri.total_amount_usd > 50000
    AND (cri.risk_flag = 'HIGH_RISK' OR cri.max_transaction_usd > 10000)
ORDER BY cri.trans_year DESC, cri.trans_month DESC, cri.total_amount_usd DESC;
```

**Mapping Notes:**
- Multiple sequential JOINs → Consolidated in single CTE with multiple JOIN clauses
- AGGREGATE components → Separate CTEs with GROUP BY
- Multiple REFORMAT steps → Multiple CTEs for transformation stages
- Complex join conditions → Additional AND clauses in JOIN ON conditions
- FILTER after aggregation → WHERE clause in final SELECT
- SORT component → ORDER BY clause
- Conditional logic in notation → CASE expressions in SQL
- Function calls (coalesce, date_format, etc.) → SQL equivalent functions

---

## Key Patterns for LLM Learning

### Pattern 1: Component to SQL Clause Mapping
- **INPUT_FILE** → `FROM` clause or `JOIN` source
- **FILTER** → `WHERE` clause (pre-aggregation) or `HAVING` clause (post-aggregation)
- **REFORMAT** → Calculated columns in `SELECT` or CTE
- **JOIN** → `JOIN` clause with `ON` condition
- **AGGREGATE** → `GROUP BY` with aggregate functions
- **SORT** → `ORDER BY` clause
- **OUTPUT_FILE** → Result set

### Pattern 2: Sequential Processing
Ab-Initio processes data sequentially through components. In SQL:
- Multiple transformations → Multiple CTEs or nested subqueries
- Each CTE represents a stage in the data pipeline
- Final SELECT combines all transformations

### Pattern 3: Join Type Mapping
- `join_type: INNER` → `INNER JOIN`
- `join_type: LEFT` → `LEFT JOIN` or `LEFT OUTER JOIN`
- `join_type: RIGHT` → `RIGHT JOIN` or `RIGHT OUTER JOIN`
- `join_type: FULL` → `FULL OUTER JOIN`

### Pattern 4: Aggregation Patterns
- `group_by` notation → `GROUP BY` clause
- `sum()`, `count()`, `avg()`, `max()`, `min()` → Aggregate functions
- Post-aggregate FILTER → `HAVING` clause
- `count(distinct field)` → `COUNT(DISTINCT column)`

### Pattern 5: Data Transformations
- Arithmetic operations → Direct SQL expressions
- Conditional logic → `CASE` statements or `IF` functions
- Date functions → SQL date/time functions
- String operations → SQL string functions
- Null handling (`coalesce`) → `COALESCE()` or `IFNULL()`

### Pattern 6: Complex Flow Decomposition
When Ab-Initio has multiple parallel or converging flows:
1. Create CTEs for each major transformation stage
2. Use descriptive CTE names that indicate the transformation purpose
3. Build final query by joining/combining CTEs
4. Maintain logical flow from source → transformation → aggregation → output

---

## Best Practices for Conversion

1. **Read the flow carefully**: Identify all input sources and their relationships
2. **Map joins systematically**: Follow join sequence from Ab-Initio graph
3. **Preserve business logic**: Ensure calculations and filters are exactly replicated
4. **Use CTEs for clarity**: Break complex logic into manageable, named steps
5. **Validate join types**: Ensure INNER/LEFT/RIGHT joins match requirements
6. **Test incrementally**: Build query step-by-step, testing each CTE
7. **Optimize performance**: Consider indexes on join keys and filter columns
8. **Document assumptions**: Note any ambiguities or interpretation decisions

---

*End of Examples Document*
