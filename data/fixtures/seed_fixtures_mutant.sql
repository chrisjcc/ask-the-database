-- ==============================================================================
-- Mutation-Augmented Seed Fixtures for AskTheDatabase Zero-Budget Eval Harness
-- Injects additional boundary conditions (unjoined records, duplicate join
-- fanout rows, NULL foreign keys, and perturbed status values) to invalidate
-- flawed queries that pass naive single-fixture set equality by coincidence.
-- ==============================================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS line_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS customers;

CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL UNIQUE,
    budget REAL,
    location TEXT
);

CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dept_id INTEGER REFERENCES departments(dept_id),
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER REFERENCES employees(emp_id),
    bonus REAL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    city TEXT,
    country TEXT NOT NULL,
    loyalty_tier TEXT,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total_amount REAL NOT NULL,
    shipping_fee REAL DEFAULT 0.0
);

CREATE TABLE line_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount REAL DEFAULT 0.0
);

-- ==============================================================================
-- Mutation Variant Records
-- ==============================================================================

-- Departments: Extra department 6 with NULL budget
INSERT INTO departments (dept_id, dept_name, budget, location) VALUES
(1, 'Engineering', 2500000.0, 'Building A'),
(2, 'Sales', 1800000.0, 'Building B'),
(3, 'Marketing', 950000.0, 'Building B'),
(4, 'Human Resources', 600000.0, 'Building C'),
(5, 'Legal', 450000.0, 'Building A'),
(6, 'Research & Development', NULL, 'Building D');

-- Employees: Added employee 111 with 0 salary and NULL dept
INSERT INTO employees (emp_id, first_name, last_name, dept_id, salary, hire_date, manager_id, bonus) VALUES
(101, 'Alice', 'Smith', 1, 185000.0, '2020-03-15', NULL, 25000.0),
(102, 'Bob', 'Jones', 1, 140000.0, '2021-06-01', 101, 0.0),
(103, 'Charlie', 'Brown', 1, 125000.0, '2022-01-10', 102, NULL),
(104, 'Diana', 'Prince', 2, 160000.0, '2019-11-20', 101, 30000.0),
(105, 'Evan', 'Wright', 2, 95000.0, '2022-04-15', 104, 12000.0),
(106, 'Fiona', 'Gallagher', 2, 65000.0, '2023-08-01', 105, NULL),
(107, 'George', 'Clark', 3, 110000.0, '2021-09-01', 101, 8000.0),
(108, 'Hannah', 'Abbott', 4, 85000.0, '2022-10-15', 101, NULL),
(109, 'Ian', 'Malcolm', NULL, 75000.0, '2024-02-01', NULL, NULL),
(110, 'Julia', 'Roberts', 1, 130000.0, '2021-12-01', 102, 5000.0),
(111, 'Kevin', 'Flynn', NULL, 90000.0, '2024-03-01', 101, NULL);

-- Customers: Customer 211 with NULL city and country UK
INSERT INTO customers (customer_id, first_name, last_name, email, city, country, loyalty_tier, signup_date) VALUES
(201, 'Arthur', 'Dent', 'arthur@galaxy.org', 'London', 'UK', 'Gold', '2023-01-15'),
(202, 'Trillian', 'Astra', 'trillian@galaxy.org', 'London', 'UK', 'Silver', '2023-02-20'),
(203, 'Ford', 'Prefect', 'ford@galaxy.org', 'New York', 'USA', 'Platinum', '2023-03-10'),
(204, 'Tricia', 'McMillan', 'tricia@subetha.com', 'Paris', 'France', NULL, '2023-04-05'),
(205, 'Zaphod', 'Beeblebrox', 'president@galaxy.gov', 'Berlin', 'Germany', 'Bronze', '2023-05-12'),
(206, 'Marvin', 'Android', 'paranoid@cyberdyne.io', 'Tokyo', 'Japan', 'Gold', '2023-06-18'),
(207, 'Slartibartfast', 'Coastline', 'fjords@magrathea.com', NULL, 'Norway', 'Silver', '2023-07-22'),
(208, 'Dirk', 'Gently', 'holistic@detective.co.uk', 'Toronto', 'Canada', NULL, '2023-08-30'),
(209, 'Eddie', 'Computer', 'shipboard@heartofgold.net', 'Sydney', 'Australia', 'Bronze', '2023-09-14'),
(210, 'Fenchurch', 'Air', 'flying@clouds.net', 'New York', 'USA', 'Gold', '2023-10-01'),
(211, 'Garth', 'Vader', 'garth@deathstar.com', NULL, 'UK', NULL, '2023-11-05');

-- Products
INSERT INTO products (product_id, product_name, category, unit_price, stock_quantity, is_active) VALUES
(301, 'Pro Laptop 15', 'Electronics', 1200.0, 45, 1),
(302, 'Wireless Optical Mouse', 'Electronics', 25.0, 150, 1),
(303, 'Mechanical RGB Keyboard', 'Electronics', 110.0, 80, 1),
(304, 'Ergonomic Desk Chair', 'Furniture', 250.0, 30, 1),
(305, 'Motorized Standing Desk', 'Furniture', 550.0, 20, 1),
(306, 'Insulated Ceramic Mug', 'Kitchen', 15.0, 200, 1),
(307, 'Stainless Steel Water Bottle', 'Kitchen', 20.0, 0, 0),
(308, 'Studio Noise-Canceling Headphones', 'Electronics', 300.0, 12, 0);

-- Orders: Order 409 added for customer 201 (another order to stress customer-order-item fanouts)
INSERT INTO orders (order_id, customer_id, order_date, status, total_amount, shipping_fee) VALUES
(401, 201, '2024-01-10', 'COMPLETED', 1250.0, 15.0),
(402, 201, '2024-05-14', 'COMPLETED', 275.0, 0.0),
(403, 202, '2024-02-18', 'COMPLETED', 135.0, 10.0),
(404, 203, '2024-03-22', 'COMPLETED', 2000.0, 25.0),
(405, 203, '2024-07-19', 'PENDING', 30.0, 5.0),
(406, 204, '2024-04-11', 'CANCELLED', 25.0, 0.0),
(407, 205, '2025-01-08', 'COMPLETED', 660.0, 20.0),
(408, 206, '2025-02-14', 'SHIPPED', 1200.0, 15.0),
(409, 201, '2025-03-01', 'COMPLETED', 25.0, 5.0);

-- Line Items
INSERT INTO line_items (item_id, order_id, product_id, quantity, unit_price, discount) VALUES
(501, 401, 301, 1, 1200.0, 0.0),
(502, 401, 302, 2, 25.0, 0.0),
(503, 402, 304, 1, 250.0, 0.0),
(504, 402, 302, 1, 25.0, 0.0),
(505, 403, 303, 1, 110.0, 0.0),
(506, 403, 302, 1, 25.0, 0.0),
(507, 404, 301, 1, 1200.0, 0.0),
(508, 404, 305, 1, 550.0, 0.0),
(509, 404, 304, 1, 250.0, 0.0),
(510, 405, 306, 2, 15.0, 0.0),
(511, 406, 302, 1, 25.0, 0.0),
(512, 407, 305, 1, 550.0, 0.0),
(513, 407, 303, 1, 110.0, 0.0),
(514, 408, 301, 1, 1200.0, 0.0),
(515, 401, 306, 1, 15.0, 0.0),
(516, 409, 302, 1, 25.0, 0.0);
