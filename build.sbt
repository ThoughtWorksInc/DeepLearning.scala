sbt.dsl.dependsOn(DifferentiableBoolean, DifferentiableDouble, DifferentiableINDArray, DifferentiableHList, DifferentiableCoproduct, DifferentiableSeq, DifferentiableAny, DifferentiableNothing)

lazy val Layer = project.disablePlugins(SparkPackagePlugin)

lazy val Lift = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val DifferentiableBoolean = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer, BufferedLayer, Poly)

lazy val DifferentiableDouble = project.disablePlugins(SparkPackagePlugin).dependsOn(Poly, DifferentiableBoolean, BufferedLayer, DifferentiableAny % Test)

lazy val Poly = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableAny = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableNothing = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableSeq = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift)

lazy val DifferentiableINDArray = project.disablePlugins(SparkPackagePlugin).dependsOn(DifferentiableDouble)

lazy val DifferentiableHList = project.disablePlugins(SparkPackagePlugin).dependsOn(Poly)

lazy val DifferentiableCoproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(DifferentiableBoolean)

lazy val BufferedLayer = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val `sbt-nd4j` = project.disablePlugins(SparkPackagePlugin)

SbtNd4J.addNd4jRuntime(Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

crossScalaVersions := Seq("2.10.6", "2.11.8")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full))

organization in ThisBuild := "com.thoughtworks.deeplearning"
