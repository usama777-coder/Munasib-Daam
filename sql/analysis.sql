-- Run with:  sqlite3 data/processed/listings.db < sql/analysis.sql   (SQLite 3.25+ for window functions)

-- 1. Median-ish price per sq ft by city (SQLite has no MEDIAN, so use the middle row)
WITH ranked AS (
  SELECT city, price_per_sqft,
         ROW_NUMBER() OVER (PARTITION BY city ORDER BY price_per_sqft) AS rn,
         COUNT(*)     OVER (PARTITION BY city) AS n
  FROM listings)
SELECT city, ROUND(AVG(price_per_sqft)) AS median_price_per_sqft, MAX(n) AS listings
FROM ranked WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
GROUP BY city ORDER BY median_price_per_sqft DESC;

-- 2. Most expensive areas per city (top 3), with at least 20 listings
WITH area AS (
  SELECT city, location, COUNT(*) AS n, AVG(price_per_sqft) AS avg_ppsf
  FROM listings GROUP BY city, location HAVING COUNT(*) >= 20),
ranked AS (
  SELECT *, RANK() OVER (PARTITION BY city ORDER BY avg_ppsf DESC) AS rnk FROM area)
SELECT city, location, n, ROUND(avg_ppsf) AS avg_price_per_sqft FROM ranked WHERE rnk <= 3 ORDER BY city, rnk;

-- 3. Month-over-month change in average price per sq ft by city
WITH m AS (
  SELECT city, listing_month, AVG(price_per_sqft) AS ppsf, COUNT(*) AS n
  FROM listings WHERE listing_month IS NOT NULL GROUP BY city, listing_month)
SELECT city, listing_month, ROUND(ppsf) AS avg_ppsf,
       ROUND(100.0 * (ppsf - LAG(ppsf) OVER (PARTITION BY city ORDER BY listing_month))
             / LAG(ppsf) OVER (PARTITION BY city ORDER BY listing_month), 1) AS mom_pct
FROM m WHERE n >= 20 ORDER BY city, listing_month;

-- 4. Price by size band (marla) and city
SELECT city,
       CASE WHEN area_sqft < 1500 THEN '<5 marla' WHEN area_sqft < 2800 THEN '5-10 marla'
            WHEN area_sqft < 5500 THEN '10 marla-1 kanal' ELSE '1 kanal+' END AS size_band,
       COUNT(*) AS n, ROUND(AVG(price_pkr) / 100000.0, 1) AS avg_price_lakh
FROM listings GROUP BY city, size_band ORDER BY city, MIN(area_sqft);
