enablePlugins(Example)

libraryDependencies += "com.thoughtworks.raii" %% "shared" % "1.0.0-M6"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test

scalacOptions += "-Ypartial-unification"
