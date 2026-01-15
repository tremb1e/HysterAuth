# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# ================== gRPC Official Rules ==================
# Keep gRPC classes
-keep class io.grpc.** { *; }
-keep class io.grpc.internal.** { *; }
-keep class io.grpc.stub.** { *; }
-keep class io.grpc.protobuf.** { *; }
-keep class io.grpc.okhttp.** { *; }

# gRPC annotations
-keepattributes *Annotation*

# gRPC generated classes
-keep class * extends io.grpc.stub.AbstractStub { *; }
-keep class * implements io.grpc.BindableService { *; }

# ================== Protobuf Official Rules ==================
# Keep Protobuf classes
-keep class com.google.protobuf.** { *; }
-keep class * extends com.google.protobuf.GeneratedMessageV3 { *; }
-keep class * extends com.google.protobuf.GeneratedMessageLite { *; }
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
    <fields>;
}
-keepclassmembers class * extends com.google.protobuf.GeneratedMessage {
    <fields>;
}

# Protobuf enums
-keepclassmembers enum * extends com.google.protobuf.Internal$EnumLite {
    public static **[] values();
    public static ** valueOf(int);
    public static ** internalGetValueMap();
}

# Keep protobuf generated classes
-keep class com.continuousauth.proto.** { *; }

# ================== Tink Official Rules ==================
# Keep Tink classes
-keep class com.google.crypto.tink.** { *; }

# Keep Tink's internal classes that might be accessed via reflection
-keep class com.google.crypto.tink.proto.** { *; }
-keep class com.google.crypto.tink.shaded.protobuf.** { *; }

# Keep Tink key managers and factories
-keep class * implements com.google.crypto.tink.KeyManager { *; }
-keep class * implements com.google.crypto.tink.KeyTypeManager { *; }
-keep class * implements com.google.crypto.tink.PrivateKeyTypeManager { *; }
-keep class * extends com.google.crypto.tink.KeyManagerImpl { *; }

# Keep classes used for Android Keystore integration
-keep class com.google.crypto.tink.integration.android.** { *; }

# Keep annotations used by Tink
-keep @interface com.google.crypto.tink.annotations.Alpha

# Don't warn about unused Tink classes
-dontwarn com.google.crypto.tink.**

# Keep classes that use @Alpha annotation
-keep @com.google.crypto.tink.annotations.Alpha class * { *; }

# ================== OkHttp and Naming Classes ==================
# Fix missing OkHttp classes (optional dependency for gRPC)
-dontwarn com.squareup.okhttp.**
-dontwarn com.squareup.okhttp3.**
-keep class com.squareup.okhttp.** { *; }
-keep class com.squareup.okhttp3.** { *; }

# Fix missing javax.naming classes (optional dependency for gRPC DNS resolution)
-dontwarn javax.naming.**
-dontwarn javax.naming.directory.**
-keep class javax.naming.** { *; }
-keep class javax.naming.directory.** { *; }

# Additional gRPC/OkHttp related rules
-dontwarn io.grpc.okhttp.internal.**
-keep class io.grpc.okhttp.internal.** { *; }

# Suppress warnings for optional dependencies
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**

# Keep Room classes
-keep class androidx.room.** { *; }
-keep class androidx.sqlite.** { *; }
-keepclassmembers class * {
    @androidx.room.* <methods>;
}

# Keep Hilt classes
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keepclasseswithmembers class * {
    @dagger.hilt.* <methods>;
}

# Keep model classes
-keep class com.continuousauth.model.** { *; }
-keep class com.continuousauth.data.** { *; }

# Keep sensor and detection classes for reflection
-keep class com.continuousauth.sensor.** { *; }
-keep class com.continuousauth.detection.** { *; }

# Keep observability classes to preserve metrics collection
-keep class com.continuousauth.observability.** { *; }

# Keep transmission and network classes
-keep class com.continuousauth.transmission.** { *; }
-keep class com.continuousauth.network.** { *; }

# Keep crypto classes - critical for security
-keep class com.continuousauth.crypto.** { *; }

# Keep core business logic classes
-keep class com.continuousauth.core.** { *; }

# Keep UI data classes and enums
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable implementations
-keep class * implements android.os.Parcelable {
    public static final android.os.Parcelable$Creator *;
}

# Keep Serializable classes
-keepclassmembers class * implements java.io.Serializable {
    static final long serialVersionUID;
    private static final java.io.ObjectStreamField[] serialPersistentFields;
    !static !transient <fields>;
    private void writeObject(java.io.ObjectOutputStream);
    private void readObject(java.io.ObjectInputStream);
    java.lang.Object writeReplace();
    java.lang.Object readResolve();
}

# Keep native method names
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep annotation for runtime reflection
-keepattributes RuntimeVisibleAnnotations
-keepattributes RuntimeInvisibleAnnotations
-keepattributes RuntimeVisibleParameterAnnotations
-keepattributes RuntimeInvisibleParameterAnnotations

# Keep source file and line numbers for crash reports while obfuscating class names
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Optimize for performance and security
-optimizationpasses 5
-optimizations !code/simplification/arithmetic,!code/simplification/cast,!field/*,!class/merging/*
-allowaccessmodification
-dontpreverify
-dontusemixedcaseclassnames
-dontskipnonpubliclibraryclasses
-verbose

# Remove debug logs in release builds for security
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int i(...);
    public static int w(...);
    public static int d(...);
    public static int e(...);
}

# Kotlin-specific rules
-keep class kotlin.** { *; }
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**
-keepclassmembers class **$WhenMappings {
    <fields>;
}
-keepclassmembers class kotlin.Metadata {
    public <methods>;
}

# Coroutines specific rules
-keep class kotlinx.coroutines.** { *; }
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}