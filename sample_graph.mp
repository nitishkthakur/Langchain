# Ab Initio Graph File - Sample ETL Pipeline
# This file contains a data processing flow with multiple components

begin graph
    name: customer_data_processing
    version: 1.0
    description: ETL pipeline for customer data processing

    # Component 1: Input File Reader
    begin component
        type: input_file
        name: read_customer_data
        file_path: /data/input/customers.dat
        layout: customer_layout.dml

        begin parameters
            out0: customer_raw_data
        end parameters
    end component

    # Component 2: Filter Component
    begin component
        type: filter_by_expression
        name: filter_active_customers
        filter_condition: status == 'ACTIVE'

        begin parameters
            in0: customer_raw_data
            out0: active_customers
            out1: inactive_customers
        end parameters
    end component

    # Component 3: Reformat Component
    begin component
        type: reformat
        name: transform_customer_data
        transform: customer_transform.xfr

        begin parameters
            in0: active_customers
            out0: transformed_customers
        end parameters
    end component

    # Component 4: Join Component
    begin component
        type: join
        name: join_with_accounts
        join_type: inner
        join_key: customer_id

        begin parameters
            in0: transformed_customers
            in1: account_data
            out0: customer_account_joined
        end parameters
    end component

    # Component 5: Account Input File
    begin component
        type: input_file
        name: read_account_data
        file_path: /data/input/accounts.dat
        layout: account_layout.dml

        begin parameters
            out0: account_data
        end parameters
    end component

    # Component 6: Aggregate Component
    begin component
        type: aggregate
        name: aggregate_by_region
        group_by: region

        begin parameters
            in0: customer_account_joined
            out0: regional_summary
        end parameters
    end component

    # Component 7: Sort Component
    begin component
        type: sort
        name: sort_by_revenue
        sort_keys: total_revenue desc

        begin parameters
            in0: regional_summary
            out0: sorted_regional_summary
        end parameters
    end component

    # Component 8: Rollup Component
    begin component
        type: rollup
        name: calculate_totals
        rollup_key: region

        begin parameters
            in0: sorted_regional_summary
            out0: rollup_output
        end parameters
    end component

    # Component 9: Output File - Main Report
    begin component
        type: output_file
        name: write_main_report
        file_path: /data/output/customer_report.dat
        layout: report_layout.dml

        begin parameters
            in0: rollup_output
        end parameters
    end component

    # Component 10: Output File - Inactive Customers
    begin component
        type: output_file
        name: write_inactive_report
        file_path: /data/output/inactive_customers.dat
        layout: customer_layout.dml

        begin parameters
            in0: inactive_customers
        end parameters
    end component

    # Component 11: Lookup Component
    begin component
        type: lookup_file
        name: lookup_customer_metadata
        lookup_key: customer_id

        begin parameters
            in0: transformed_customers
            in1: metadata_reference
            out0: enriched_customers
            out1: unmatched_customers
        end parameters
    end component

    # Component 12: Metadata Reference Input
    begin component
        type: input_file
        name: read_metadata
        file_path: /data/reference/metadata.dat
        layout: metadata_layout.dml

        begin parameters
            out0: metadata_reference
        end parameters
    end component

    # Component 13: Output File - Unmatched Customers
    begin component
        type: output_file
        name: write_unmatched_report
        file_path: /data/output/unmatched_customers.dat
        layout: customer_layout.dml

        begin parameters
            in0: unmatched_customers
        end parameters
    end component

    # Flow Definitions
    begin flows
        flow: customer_raw_data from read_customer_data to filter_active_customers
        flow: active_customers from filter_active_customers to transform_customer_data
        flow: inactive_customers from filter_active_customers to write_inactive_report
        flow: transformed_customers from transform_customer_data to join_with_accounts
        flow: transformed_customers from transform_customer_data to lookup_customer_metadata
        flow: account_data from read_account_data to join_with_accounts
        flow: metadata_reference from read_metadata to lookup_customer_metadata
        flow: customer_account_joined from join_with_accounts to aggregate_by_region
        flow: enriched_customers from lookup_customer_metadata to join_with_accounts
        flow: unmatched_customers from lookup_customer_metadata to write_unmatched_report
        flow: regional_summary from aggregate_by_region to sort_by_revenue
        flow: sorted_regional_summary from sort_by_revenue to calculate_totals
        flow: rollup_output from calculate_totals to write_main_report
    end flows

end graph
