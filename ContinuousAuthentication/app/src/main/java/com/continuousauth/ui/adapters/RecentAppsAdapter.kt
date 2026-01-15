package com.continuousauth.ui.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.continuousauth.databinding.ItemRecentAppBinding
import com.continuousauth.ui.viewmodels.SensorDataViewModel
import java.text.SimpleDateFormat
import java.util.*

/**
 * Adapter for the recent apps list.
 */
class RecentAppsAdapter : ListAdapter<SensorDataViewModel.RecentApp, RecentAppsAdapter.AppViewHolder>(AppDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): AppViewHolder {
        val binding = ItemRecentAppBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return AppViewHolder(binding)
    }

    override fun onBindViewHolder(holder: AppViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    /**
     * ViewHolder
     */
    class AppViewHolder(
        private val binding: ItemRecentAppBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

        fun bind(app: SensorDataViewModel.RecentApp) {
            // Display app name.
            binding.tvAppName.text = app.appName.ifEmpty { app.packageName }
            
            // Display package name.
            binding.tvPackageName.text = app.packageName
            
            // Display timestamp.
            binding.tvTimestamp.text = timeFormat.format(Date(app.timestamp))
        }
    }

    /**
     * DiffCallback
     */
    private class AppDiffCallback : DiffUtil.ItemCallback<SensorDataViewModel.RecentApp>() {
        override fun areItemsTheSame(
            oldItem: SensorDataViewModel.RecentApp,
            newItem: SensorDataViewModel.RecentApp
        ): Boolean {
            return oldItem.packageName == newItem.packageName && 
                   oldItem.timestamp == newItem.timestamp
        }

        override fun areContentsTheSame(
            oldItem: SensorDataViewModel.RecentApp,
            newItem: SensorDataViewModel.RecentApp
        ): Boolean {
            return oldItem == newItem
        }
    }
}
