with source as (
    select *
    from {{ source('raw', 'raw_tickets') }}
),

renamed as (
    select
        cast(ticket_id as varchar) as ticket_id,
        cast(created_at as timestamp) as created_at,

        cast(product as varchar) as product,
        cast(product_module as varchar) as product_module,
        cast(channel as varchar) as channel,
        cast(customer_tier as varchar) as customer_tier,
        cast(environment as varchar) as environment,
        cast(language as varchar) as language,
        cast(region as varchar) as region,

        cast(priority as varchar) as priority,
        cast(severity as varchar) as severity,

        cast(subject as varchar) as subject,
        cast(description as varchar) as description,
        cast(error_logs as varchar) as error_logs,
        cast(stack_trace as varchar) as stack_trace,
        cast(customer_sentiment as varchar) as customer_sentiment,
        cast(feedback_text as varchar) as feedback_text,

        cast(account_age_days as integer) as account_age_days,
        cast(account_monthly_value as double) as account_monthly_value,
        cast(previous_tickets as integer) as previous_tickets,
        cast(product_version_age_days as integer) as product_version_age_days,
        cast(attachments_count as integer) as attachments_count,
        cast(ticket_text_length as integer) as ticket_text_length,

        cast(category as varchar) as category,
        cast(subcategory as varchar) as subcategory
)

select * from renamed
