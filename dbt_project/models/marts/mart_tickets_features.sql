with stg as (
    select * from {{ ref('stg_tickets') }}
)

select
    ticket_id,
    created_at,
    category,
    subcategory,
    subject,
    description,

    -- Extract error code from description using regex
    -- Pattern: ERROR_[A-Z0-9_]+ (e.g., ERROR_AUTH_401, ERROR_TIMEOUT_429)
    coalesce(
        regexp_extract(description, 'ERROR_[A-Z0-9_]+', 0),
        'NO_ERROR'
    ) as error_code,

    -- Template ID: hash of (subject, description) for grouping
    md5(subject || '|||' || description) as template_id,

    -- Customer fields
    product,
    product_module,
    channel,
    customer_tier,
    environment,
    language,
 region,
    priority,
    severity,

    -- Numeric fields
    account_age_days,
    account_monthly_value,
    previous_tickets,
    product_version_age_days,
    attachments_count,
    ticket_text_length,

    -- Text fields
    error_logs,
    stack_trace,

    -- Sentiment analysis columns (post-resolution feedback)
    customer_sentiment,
    feedback_text

from stg
